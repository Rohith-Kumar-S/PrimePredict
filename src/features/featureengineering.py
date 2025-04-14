import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, data, holidays, holidays_2021, inflation, events):
        self.data = data
        self.holidays = holidays
        self.holidays_2021 = holidays_2021
        self.inflation = inflation
        self.events = events
        self.add_total_sales()
        self.df = data.groupby(data.index)["total_sales"].sum()
        self.df = self.generate_time_features(self.df)
        self.df = self.df[self.df["year"] < 2023]
        self.add_amazon_events()
        self.add_holidays()
        self.add_states()
        self.add_previous_sales()
        self.add_inflation()

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

    def add_previous_sales(self):
        self.df["Sales 1YA"] = self.assign_historic_sales(self.df, year_till=2021)
        self.df["Sales 2YA"] = self.assign_historic_sales(self.df, year_till=2020)
        self.df["Sales 3YA"] = self.assign_historic_sales(self.df, year_till=2019)

    def add_amazon_events(self):
        self.df = pd.merge(
            self.df, self.events, left_index=True, right_index=True, how="left"
        )

        self.df["Amazon Events"] = self.df["Amazon Events"].fillna("No Events")
        self.df = pd.get_dummies(self.df, drop_first=True)
        if (
            not pd.Series(self.df.index.year.isin([2024, 2025, 2026]))
            .value_counts()
            .index[0]
        ):
            self.df["Amazon Events_Big Spring Sale"] = False

    def add_inflation(self):
        self.inflation["observation_date"] = pd.to_datetime(
            self.inflation["observation_date"]
        )
        self.inflation = self.inflation.rename(
            columns={"T10YIEM": "inflation_rate"}
        ).set_index("observation_date")

        self.df = pd.merge(
            self.df, self.inflation, left_index=True, right_index=True, how="left"
        )

        self.df["inflation_rate"] = self.df["inflation_rate"].interpolate()

    def add_states(self):
        states_list = list(
            self.data[self.data.index.year < 2023]["Shipping Address State"]
            .dropna().sort_values()
            .unique()
        )
        statesdf = pd.DataFrame(
            data=dict(zip(range(len(states_list)), [0] * len(states_list))),
            index=range(len(states_list)),
        )
        statesdf = statesdf.iloc[0:1]

        def add_state_features(state_included_by_date):
            nonlocal statesdf
            statesdf = pd.concat(
                [
                    statesdf,
                    pd.DataFrame(
                        pd.Series(states_list)
                        .map(
                            dict(
                                zip(
                                    list(state_included_by_date), [1] * len(states_list)
                                )
                            )
                        )
                        .fillna(0)
                        .astype(int)
                    ).transpose(),
                ],
                axis=0,
            )

        np.vectorize(lambda states_list: add_state_features(states_list))(
            self.data[self.data.index.year < 2023]
            .groupby(["Order Date"])["Shipping Address State"]
            .unique()
        )
        statesdf.index = range(len(statesdf.index))
        statesdf = statesdf.iloc[2:]
        statesdf.index = range(len(statesdf.index))
        statesdf.columns = states_list
        
        self.df = pd.merge(
            self.df.reset_index(),
            statesdf,
            left_index=True,
            right_index=True,
            how="left",
        ).set_index("Order Date")

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

    def generate_time_features(self, df):
        df = df.reset_index()
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
        return df

    def output(self):
        return self.df
