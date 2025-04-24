from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class SalesForecaster:
    """SalesForecaster is a class that handles the training and prediction of sales data using XGBoost and CatBoost models.
    It also provides methods for cross-validation and feature importance extraction.
    """

    def __init__(self, entity_name=""):
        """__init__ initializes the SalesForecaster class with empty models list and loads pre-trained models if available.
        It also sets the entity name for the model, if provided.

        Args:
            entity_name (str, optional): state or category to forecast. Defaults to ''.
        """

        self.models = []
        self.models.append(xgb.XGBRegressor())
        self.models.append(CatBoostRegressor())
        models_path = os.path.join(os.getcwd(), "models", "saves")
        if not "src" in models_path:
            models_path = os.path.join(os.getcwd(), "src", "models", "saves")
        if not entity_name == "":
            entity_name = f"{entity_name}_"
        try:
            self.models[0].load_model(
                os.path.join(models_path, f"{entity_name}xgb.json")
            )
            self.models[1].load_model(
                os.path.join(models_path, f"{entity_name}cat.cbm")
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.models = []

    def is_trained(self):
        """is_trained checks if the model has been trained by checking if the models list is empty.
        If the list is not empty, it means the model has been trained and loaded successfully.

        Returns:
            _type_: bool: True if the model is trained, False otherwise.
        """
        return not len(self.models) == 0

    def get_feature_columns(self, df):
        """get_feature_columns returns the feature columns from the dataframe excluding the target variable and other specified columns.

        Returns:
            _type_: list: list of feature columns to be used for training the model.
        """

        salesdf_columns = list(df.columns)
        [salesdf_columns.remove(column) for column in ["total_sales", "S1", "S2", "S3"]]
        return sorted(salesdf_columns)

    def cross_validate(self, df, forcastdf, salesdf_columns):
        """cross_validate performs cross-validation on the given dataframe using TimeSeriesSplit.

        Args:
            df (_type_): train data
            forcastdf (_type_): forecaast data
            salesdf_columns (_type_): feature columns to be used for training the model

        Returns:
            _type_: tuple: predictions, scores, and feature importances from the cross-validation.
        """

        scores = []
        preds = []
        feature_importances = []
        tss = TimeSeriesSplit(n_splits=3, test_size=365, gap=0)
        salesdf_columns = self.get_feature_columns(df)
        for train_idx, val_idx in tss.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[val_idx]
            X_train = train[salesdf_columns]
            y_train = train["total_sales"]
            X_test = test[salesdf_columns]
            y_test = test["total_sales"]

            reg_model = xgb.XGBRegressor(
                base_score=0.5,
                booster="gblinear",
                n_estimators=1000,
                early_stopping_rounds=50,
                objective="reg:squarederror",
                learning_rate=0.01,
            )
            reg_model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )

            feature_importances.append(reg_model.feature_importances_)
            y_preds = reg_model.predict(X_train)
            ds = pd.concat([y_train.reset_index(), pd.Series(y_preds)], axis=1)
            ds = ds.set_index("Order Date")
            ds["total_sales"].plot(label="true data")
            ds[0].plot(label="prediction")
            plt.legend()
            plt.show()
            y_preds = reg_model.predict(X_test)
            ds = pd.concat([y_test.reset_index(), pd.Series(y_preds)], axis=1)
            ds = ds.set_index("Order Date")
            ds["total_sales"].plot(label="true data")
            ds[0].plot(label="prediction")
            plt.legend()
            plt.show()
            forcast_preds = reg_model.predict(forcastdf)
            ds = pd.Series(forcast_preds, index=forcastdf.index)
            ds.plot(label="prediction")
            plt.legend()
            plt.show()
            preds.append(y_preds)
            score = np.sqrt(mean_squared_error(y_test, y_preds))
            scores.append(score)

        return preds, scores, feature_importances

    def train(self, df, entity_name=""):
        """train trains the XGBoost and CatBoost models on the given dataframe.
        It also saves the trained models to disk.

        Args:
            df (_type_): full dataframe to train the model on.
            entity_name (str, optional): state or category to forecast. Defaults to ''.
        """

        salesdf_columns = self.get_feature_columns(df)
        X_train = df[salesdf_columns]
        y_train = df["total_sales"]

        xgb_model = xgb.XGBRegressor(
            base_score=0.5,
            booster="gblinear",
            n_estimators=400,
            early_stopping_rounds=50,
            objective="reg:squarederror",
            learning_rate=0.01,
            reg_alph= 0, 
            reg_lambda= 0
        )

        cat_model = CatBoostRegressor(
            depth=3,
            l2_leaf_reg=2,
            learning_rate=0.01,
            n_estimators=1000,
            silent=True,
        )

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train[salesdf_columns], y_train)],
            verbose=False,
        )

        cat_model.fit(
            X_train,
            y_train,
            early_stopping_rounds=50,
            verbose=False,
        )
        self.models.append(xgb_model)
        self.models.append(cat_model)
        save_file_name = "" if entity_name == "" else f"{entity_name}_"
        save_path = os.path.join(os.getcwd(), "models", "saves")
        if not "src" in save_path:
            save_path = os.path.join(os.getcwd(), "src", "models", "saves")
        print("saves: ", os.path.join(save_path, f"{save_file_name}xgb.json"))
        xgb_model.save_model(os.path.join(save_path, f"{save_file_name}xgb.json"))
        cat_model.save_model(os.path.join(save_path, f"{save_file_name}cat.cbm"))

    def predict(self, df):
        """predict predicts the sales using the trained models on the given dataframe.
        It returns the predictions from both models.

        Args:
            df (_type_): forecast data to predict sales.

        Returns:
            _type_: tuple: predictions from XGBoost and CatBoost models.
        """
        salesdf_columns = sorted(df.columns)
        X_test = df[salesdf_columns]

        xgb_preds = self.models[0].predict(X_test)
        cat_preds = self.models[1].predict(X_test)
        return xgb_preds, cat_preds

    def get_feature_importance(self):
        """get_feature_importance returns the feature importances from both models.

        Raises:
            ValueError: if the model has not been trained yet.

        Returns:
            _type_: tuple: feature importances from XGBoost and CatBoost models.
        """

        if len(self.models) == 0:
            raise ValueError("Model has not been trained yet.")
        return self.model[0].feature_importances_, self.model[1].feature_importances_
