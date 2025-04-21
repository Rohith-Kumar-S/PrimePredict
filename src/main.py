from preproccessing.datapreprocessing import DataPreprocessor
from features.featureengineering import FeatureEngineering
from data.dataloader import DataLoader
import pandas as pd
from models.model import Model


def forcast(start_date, end_date, inflation_rate):

    cross_validate = False

    if Model().is_trained():
        print("Model already trained.")
        data = DataLoader(is_training=False)
        model = Model()
        forcast_df = FeatureEngineering(
            DataPreprocessor(pd.read_csv("processed_data.csv")).output(),
            data.holidays,
            data.holidays_past_2021,
            data.amazon_events,
            is_train=False,
            start=start_date,
            end=end_date,
        ).output()
        xgb_preds, cat_preds = model.predict(forcast_df)
        return pd.DataFrame(
            {
                "dates": forcast_df.index,
                "xgb predictions": xgb_preds,
                "cat predictions": cat_preds,
            }
        )

    else:
        print("Model not trained. Training now...")
        data = DataLoader(is_training=True)
        preprocessed_data = DataPreprocessor(
            data.purchases, data.products, data.categories
        ).output()

        df = FeatureEngineering(
            preprocessed_data,
            data.holidays,
            data.holidays_past_2021,
            data.amazon_events
        ).output()

        df.reset_index(inplace=True)
        df.to_csv("processed_data.csv", index=False)
        print("Data preprocessed and saved to processed_data.csv")
        df.set_index("Order Date", inplace=True)

        if cross_validate:
            forcast_df = FeatureEngineering(
                DataPreprocessor(pd.read_csv("processed_data.csv")).output(),
                data.holidays,
                data.holidays_past_2021,
                data.amazon_events,
                is_train=False,
                start="2023-01-01",
                end="2023-12-31",
            ).output()
            Model().cross_validate(df, forcast_df, forcast_df.columns)

        Model().train(df)
        print("Model trained and saved to model.json")


if __name__ == "__main__":
    print(forcast("2023-01-01", "2023-12-31", 0.03))
