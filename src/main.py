from data.dataloader import DataLoader
from preproccessing.datapreprocessing import DataPreprocessor
from features.featureengineering import FeatureEngineering
import pandas as pd
from models.model import Model


def main():

    cross_validate = False
    data = DataLoader()

    if Model(model_path="model.json").is_trained():
        print("Model already trained.")
        model = Model(model_path="model.json")
        forcast_df = FeatureEngineering(
            DataPreprocessor(pd.read_csv("processed_data.csv")).output(),
            data.holidays,
            data.holidays_past_2021,
            data.inflation,
            data.amazon_events,
            data.get_usa_states(),
            is_train=False,
            start="2023-01-01",
            end="2023-12-31",
        ).output()
        y_preds = model.predict(forcast_df)
        print(y_preds)
    else:
        preprocessed_data = DataPreprocessor(data.purchases).output()
        df = FeatureEngineering(
            preprocessed_data,
            data.holidays,
            data.holidays_past_2021,
            data.inflation,
            data.amazon_events,
            data.get_usa_states(),
        ).output()

        df.reset_index(inplace=True)
        df.to_csv("processed_data.csv", index=False)
        print("Data preprocessed and saved to processed_data.csv")
        df.set_index('Order Date', inplace=True)

        if cross_validate:
            forcast_df = FeatureEngineering(
                pd.read_csv("processed_data.csv"),
                data.holidays,
                data.holidays_past_2021,
                data.inflation,
                data.amazon_events,
                data.get_usa_states(),
                is_train=False,
                start="2023-01-01",
                end="2023-12-31",
            ).output()
            Model().cross_validate(df, forcast_df, forcast_df.columns)

        Model().train(df)
        print("Model trained and saved to model.json")


if __name__ == "__main__":
    main()
