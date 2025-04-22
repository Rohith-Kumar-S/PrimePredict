from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class SalesForecaster:

    def __init__(self, entity_name=""):
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
        return not len(self.models) == 0

    def get_feature_columns(self, df):
        salesdf_columns = list(df.columns)
        [salesdf_columns.remove(column) for column in ["total_sales", "S1", "S2", "S3"]]
        return sorted(salesdf_columns)

    def cross_validate(self, df, forcastdf, salesdf_columns):
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
        )
        n_estimators = 800 if entity_name == "CA" else 1000

        cat_model = CatBoostRegressor(
            depth=3,
            l2_leaf_reg=2,
            learning_rate=0.01,
            n_estimators=n_estimators,
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
        salesdf_columns = sorted(df.columns)
        X_test = df[salesdf_columns]

        xgb_preds = self.models[0].predict(X_test)
        cat_preds = self.models[1].predict(X_test)
        return xgb_preds, cat_preds

    def get_feature_importance(self):
        if len(self.models) == 0:
            raise ValueError("Model has not been trained yet.")
        return self.model[0].feature_importances_, self.model[1].feature_importances_
