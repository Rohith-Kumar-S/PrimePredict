from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Model:

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = xgb.XGBRegressor()
            try:
                self.model.load_model(model_path)
            except xgb.core.XGBoostError as e:
                print(f"Error loading model: {e}")
                self.model = None
            
    def is_trained(self):
        return self.model is not None

    def get_feature_columns(self, df):
        salesdf_columns = list(df.columns)
        [
            salesdf_columns.remove(column)
            for column in ["total_sales", "SS1", "SS2", "SS3"]
        ]
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

    def train(self, df):
        salesdf_columns = self.get_feature_columns(df)
        X_train = df[salesdf_columns]
        y_train = df["total_sales"]
        reg_model = xgb.XGBRegressor(
            base_score=0.5,
            booster="gblinear",
            n_estimators=400,
            early_stopping_rounds=50,
            objective="reg:squarederror",
            learning_rate=0.01,
        )
        reg_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train[salesdf_columns], y_train)],
            verbose=False,
        )
        self.model = reg_model
        reg_model.save_model("model.json")

    def predict(self, df):
        salesdf_columns = sorted(df.columns)
        X_test = df[salesdf_columns]
        y_preds = self.model.predict(X_test)
        return y_preds
    
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.feature_importances_
