import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class FeatureEngineering:
    def __init__(
        self,
        data,
        holidays,
        holidays_2021,
        events,
        start=None,
        end=None,
        is_train=True,
    ):
        self.data = data
        self.holidays = holidays
        self.holidays_2021 = holidays_2021
        self.events = events
        self.is_train = is_train
        if is_train:
            self.add_total_sales()
            self.df = data.groupby(data.index)["total_sales"].sum()
        else:
            self.df = pd.DataFrame(pd.date_range(start, end), columns=["Order Date"])

        self.generate_time_features()
        self.add_amazon_events()
        self.add_holidays()
        self.add_previous_sales()

    def add_holidays(self):
        self.holidays["Date"] = pd.to_datetime(self.holidays["Date"])
        self.holidays["is_holiday"] = True
        self.df = pd.merge(
            self.df,
            self.holidays[["Date", "is_holiday"]].set_index("Date"),
            how="left",
            left_index=True,
            right_index=True,
        )

        self.df = pd.merge(
            self.df,
            self.holidays_2021,
            how="left",
            left_index=True,
            right_index=True,
        )

        self.df["fedral_holiday"] = self.df["is_holiday"].fillna(False) + self.df[
            "holiday"
        ].fillna(False)

        self.df = self.df.drop(["is_holiday", "holiday"], axis=1)

    def add_total_sales(self):
        # Example feature engineering: adding a new feature based on existing ones
        self.data["total_sales"] = (
            self.data["Purchase Price Per Unit"] * self.data["Quantity"]
        )

    def get_feature_columns(self):
        salesdf_columns = list(self.df.columns)
        if not self.is_train:
            salesdf_columns = list(self.data.columns)
            [
                salesdf_columns.remove(column)
                for column in ["total_sales", "SS1", "SS2", "SS3"]
            ]
        return sorted(salesdf_columns)

    def add_previous_sales(self):
        if self.is_train:
            self.df["Sales 1YA"] = self.assign_historic_sales(self.df, year_till=2021)
            self.df["Sales 2YA"] = self.assign_historic_sales(self.df, year_till=2020)
            self.df["Sales 3YA"] = self.assign_historic_sales(self.df, year_till=2019)
            self.add_states_lags()
        else:
            combined_df = self.data.copy()
            self.df["forcasting"] = True
            combined_df["forcasting"] = False

            combined_df = pd.concat([combined_df, self.df], axis=0)

            combined_df["Sales 1YA"] = self.assign_historic_sales(
                combined_df, year_till=2022
            )
            combined_df["Sales 2YA"] = self.assign_historic_sales(
                combined_df, year_till=2021
            )
            combined_df["Sales 3YA"] = self.assign_historic_sales(
                combined_df, year_till=2020
            )

            combined_df[["SS1 1YA", "SS2 1YA", "SS3 1YA"]] = combined_df[
                ["SS1", "SS2", "SS3"]
            ].shift(self.get_shift_value(2022, is_forcast=True))
            combined_df[["SS1 2YA", "SS2 2YA", "SS3 2YA"]] = combined_df[
                ["SS1", "SS2", "SS3"]
            ].shift(self.get_shift_value(2021, is_forcast=True))
            combined_df[["SS1 3YA", "SS2 3YA", "SS3 3YA"]] = combined_df[
                ["SS1", "SS2", "SS3"]
            ].shift(self.get_shift_value(2020, is_forcast=True))

            self.df[
                [
                    "Sales 1YA",
                    "Sales 2YA",
                    "Sales 3YA",
                    "SS1 1YA",
                    "SS2 1YA",
                    "SS3 1YA",
                    "SS1 2YA",
                    "SS2 2YA",
                    "SS3 2YA",
                    "SS1 3YA",
                    "SS2 3YA",
                    "SS3 3YA",
                ]
            ] = combined_df[combined_df["forcasting"] == True][
                [
                    "Sales 1YA",
                    "Sales 2YA",
                    "Sales 3YA",
                    "SS1 1YA",
                    "SS2 1YA",
                    "SS3 1YA",
                    "SS1 2YA",
                    "SS2 2YA",
                    "SS3 2YA",
                    "SS1 3YA",
                    "SS2 3YA",
                    "SS3 3YA",
                ]
            ]

            self.df = self.df.drop("forcasting", axis=1)
        self.df = self.df.reindex(self.get_feature_columns(), axis=1)

    def add_amazon_events(self):
        self.df = pd.merge(
            self.df, self.events, left_index=True, right_index=True, how="left"
        )

        self.df["Amazon Events"] = self.df["Amazon Events"].fillna("No Events")
        self.df = pd.get_dummies(self.df, drop_first=True)

    def assign_historic_sales(self, df, year_till=2022):
        df = df.reset_index()
        sales = df[["Order Date", "total_sales"]]
        sales = sales.set_index("Order Date")
        past_sales = list(sales[sales.index.year <= year_till]["total_sales"])
        lag = [np.nan] * (len(sales) - len(past_sales))
        lag.extend(past_sales)
        lag = pd.Series(lag, index=sales.index)
        df = df.set_index("Order Date")
        return lag

    def generate_time_features(self):
        df = self.df.reset_index()
        df.loc[:, "day"] = df["Order Date"].dt.day
        df.loc[:, "month"] = df["Order Date"].dt.month
        df.loc[:, "year"] = df["Order Date"].dt.year
        df.loc[:, "is_weekend"] = df["Order Date"].dt.weekday > 5
        df.loc[:, "day_of_week"] = df["Order Date"].dt.day_of_week
        df.loc[:, "day_of_year"] = df["Order Date"].dt.day_of_year
        df.loc[:, "quarter"] = df["Order Date"].dt.quarter
        df.loc[:, "is_month_start"] = df["Order Date"].dt.is_month_start
        df.loc[:, "is_month_end"] = df["Order Date"].dt.is_month_end
        df.loc[:, "is_year_start"] = df["Order Date"].dt.is_year_start
        df.loc[:, "is_year_end"] = df["Order Date"].dt.is_year_end
        df = df.set_index("Order Date")
        if self.is_train:
            self.df = df[df["year"] < 2023]
        else:
            self.df = df

    def get_shift_value(self, year, is_forcast=False):

        if not is_forcast:
            return (
                self.df[self.df.index.year < 2023].index[-1]
                - self.df[self.df.index.year <= year].index[-1]
            ).days - 10
        else:
            return (
                self.df.index[-1] - self.data[self.data.index.year <= year].index[-1]
            ).days - 10

    def add_states_lags(self):

        temp_df = self.data[
            ["Shipping Address State", "Category", "total_sales"]
        ].copy()

        temp_df = pd.DataFrame(
            temp_df.groupby([temp_df.index, "Shipping Address State"])[
                "total_sales"
            ].sum()
        ).reset_index()

        states_features = (
            pd.pivot(
                temp_df[["Order Date", "Shipping Address State", "total_sales"]],
                index="Order Date",
                columns="Shipping Address State",
                values="total_sales",
            )
            .reset_index()
            .fillna(0)
        )

        states_features = states_features.set_index("Order Date")

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(states_features)

        pca = PCA(n_components=3, random_state=101)

        pca_features = pca.fit_transform(scaled_data)

        state_sales_reduced = pd.DataFrame(pca_features)

        state_sales_reduced.rename(columns={0: "SS1", 1: "SS2", 2: "SS3"}, inplace=True)

        state_sales_reduced.set_index(self.df.index, inplace=True)

        states_features = pd.concat(
            [
                state_sales_reduced,
                state_sales_reduced.shift(self.get_shift_value(2021)).rename(
                    columns={"SS1": "SS1 1YA", "SS2": "SS2 1YA", "SS3": "SS3 1YA"}
                ),
                state_sales_reduced.shift(self.get_shift_value(2020)).rename(
                    columns={"SS1": "SS1 2YA", "SS2": "SS2 2YA", "SS3": "SS3 2YA"}
                ),
                state_sales_reduced.shift(self.get_shift_value(2019)).rename(
                    columns={"SS1": "SS1 3YA", "SS2": "SS2 3YA", "SS3": "SS3 3YA"}
                ),
            ],
            axis=1,
        )

        self.df = pd.merge(
            self.df, states_features, left_index=True, right_index=True, how="left"
        )

    def output(self):
        return self.df
