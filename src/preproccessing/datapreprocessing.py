import pandas as pd


class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.set_index()
        if "Shipping Address State" in df.columns:
            self.remove_null_values()

    def set_index(self):
        self.df["Order Date"] = pd.to_datetime(self.df["Order Date"])
        self.df.set_index("Order Date", inplace=True)
        self.df = self.df[self.df.index.year<2023]
        
    def remove_null_values(self):
        # Drop unnecessary columns
        self.df = self.df[
            self.df["Shipping Address State"].notnull()
        ]

    def output(self):
        # Output the processed data
        return self.df
