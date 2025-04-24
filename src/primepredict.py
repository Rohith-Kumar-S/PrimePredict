from preproccessing.datapreprocessing import DataPreprocessor
from features.featureengineering import FeatureEngineering
from data.dataloader import DataLoader
import pandas as pd
from models.forecast import SalesForecaster
import os


class PrimePredict:

    def __init__(self):
        """PrimePredict is a class that handles the sales forecasting process.
        It includes methods for preparing training data, processing results, and getting states and categories by frequency.
        """
        pass

    def prepare_train_data(self, data, entity_name, is_state):
        """prepare_train_data prepares the data for training by preprocessing and feature engineering.
        It saves the processed data to a CSV file in the processed_datasets directory.

        Args:
            data (_type_): data object containing purchases, products, categories, holidays_past_2021, and amazon_events
            entity_name (_type_): state or category name to forecast
            is_state (bool): whether the entity to forecast is a state or a category

        Returns:
            _type_: DataFrame: processed data ready for training
        """

        processed_data_path = os.path.join(os.getcwd(), "data", "processed_datasets")
        if not "src" in processed_data_path:
            processed_data_path = os.path.join(
                os.getcwd(), "src", "data", "processed_datasets"
            )

        preprocessed_data = DataPreprocessor(
            data.purchases,
            data.products,
            data.categories,
            entity_to_forcast=entity_name,
            is_state=is_state,
        ).output()

        df = FeatureEngineering(
            preprocessed_data,
            data.holidays_past_2021,
            data.amazon_events,
            holidays=data.holidays,
            state_forcast=is_state,
        ).output()

        df.reset_index(inplace=True)
        if entity_name == "":
            df.to_csv(
                os.path.join(processed_data_path, "overall_sales.csv"), index=False
            )
        else:
            df.to_csv(
                os.path.join(processed_data_path, f"{entity_name}_sales.csv"),
                index=False,
            )
        print("Data preprocessed and saved")
        df.set_index("Order Date", inplace=True)
        return df

    def process_results(
        self,
        start_year,
        end_year,
        forcast_df,
        xgb_preds,
        cat_preds,
        overall_sales,
    ):
        """process_results processes the results of the forecast by creating a DataFrame with the predictions and the actual sales.

        Args:
            start_year (_type_): Start year of the forecast
            end_year (_type_): End year of the forecast
            forcast_df (_type_): DataFrame containing the forecasted data
            xgb_preds (_type_): predictions from the XGBoost model
            cat_preds (_type_): predictions from the CatBoost model
            overall_sales (_type_): overall sales data

        Returns:
            _type_: tuple: DataFrame with the predictions and actual sales, and the number of years of data
        """
        print("Processing results...")

        return (
            pd.DataFrame(
                {
                    "dates": forcast_df.index,
                    "Sales Prediction - xbg": xgb_preds,
                    "Sales Prediction - cat": cat_preds,
                }
            ),
            overall_sales[
                overall_sales.index.year > (2021 - int(end_year - start_year))
            ]["total_sales"],
            int(end_year - start_year),
        )

    def get_state_and_categories_by_frequency(self, data):
        """get_state_and_categories_by_frequency gets the states and categories by frequency of purchases.

        Args:
            data (_type_): data object containing purchases, products, categories

        Returns:
            _type_: tuple: list of states and categories ordered by frequency
        """
        preprocessed_data = DataPreprocessor(
            data.purchases, data.products, data.categories
        ).output()
        states_frequencies = (
            preprocessed_data.groupby(
                [preprocessed_data.index, "Shipping Address State"]
            )["total_sales"]
            .sum()
            .reset_index()["Shipping Address State"]
            .value_counts()
            .sort_values(ascending=False)
            .index
        )
        category_frequencies = (
            preprocessed_data.groupby([preprocessed_data.index, "Category"])[
                "total_sales"
            ]
            .sum()
            .reset_index()["Category"]
            .value_counts()
            .sort_values(ascending=False)
            .index
        )
        return list(states_frequencies), list(category_frequencies)

    def forcast(self, start_date, end_date, data, entity_name="", is_state=None):
        """forcast function to forecast sales for a given date range and entity (state or category).

        Args:
            start_date (_type_): start date for the forecast
            end_date (_type_): end date for the forecast
            data (_type_): datafrom DataLoader
            entity_name (str, optional): state or category Defaults to "".
            is_state (_type_, optional): whether the entity to forecast is a state or a category, Defaults to None.

        Returns:
            _type_: tuple: DataFrame with the predictions and actual sales, and the number of years of data
        """

        cross_validate = False
        processed_data_path = os.path.join(os.getcwd(), "data", "processed_datasets")
        if not "src" in processed_data_path:
            processed_data_path = os.path.join(
                os.getcwd(), "src", "data", "processed_datasets"
            )
        if SalesForecaster(entity_name).is_trained():
            print("Model already trained.")
            data = DataLoader(is_training=False)
            model = SalesForecaster(entity_name)
            if entity_name == "":
                overall_sales = DataPreprocessor(
                    pd.read_csv(os.path.join(processed_data_path, "overall_sales.csv"))
                ).output()
            else:
                overall_sales = DataPreprocessor(
                    pd.read_csv(
                        os.path.join(processed_data_path, f"{entity_name}_sales.csv")
                    )
                ).output()
            forcast_df = FeatureEngineering(
                overall_sales,
                data.holidays_past_2021,
                data.amazon_events,
                is_train=False,
                start=start_date,
                end=end_date,
            ).output()
            xgb_preds, cat_preds = model.predict(forcast_df)
            # ca_xgb_preds, ca_cat_preds = model.predict(ca_forcast_df, state_name="CA")
            # ga_xgb_preds, ga_cat_preds = model.predict(ga_forcast_df, state_name="GA")
            return self.process_results(
                pd.to_datetime(start_date).year,
                pd.to_datetime(end_date).year,
                forcast_df,
                xgb_preds,
                cat_preds,
                overall_sales,
            )

        else:
            print("Model not trained. Training now...")
            df = self.prepare_train_data(data, entity_name, is_state)

            if cross_validate:
                forcast_df = FeatureEngineering(
                    DataPreprocessor(pd.read_csv("overall_sales.csv")).output(),
                    data.holidays_past_2021,
                    data.amazon_events,
                    is_train=False,
                    start="2023-01-01",
                    end="2023-12-31",
                ).output()
                SalesForecaster().cross_validate(df, forcast_df, forcast_df.columns)

            SalesForecaster(entity_name).train(df, entity_name=entity_name)
            print("Model trained and saved to model.json")
            return None, None, None


if __name__ == "__main__":
    # print(forcast("2023-01-01", "2023-12-31"))
    overall_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04", "2024-04-04", DataLoader(is_training=True)
    )

    statewise_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04",
        "2024-04-04",
        DataLoader(is_training=True),
        "CA",
        is_state=True,
    )

    categorywise_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04",
        "2024-04-04",
        DataLoader(is_training=True),
        "ABIS_BOOK",
        is_state=False,
    )

    print(overall_prediction_df)
    print(statewise_prediction_df)
    print(categorywise_prediction_df)
