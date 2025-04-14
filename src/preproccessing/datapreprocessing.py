import pandas as pd


class DataPreprocessor:
    def __init__(self, purchases, categories, products):
        self.purchases = purchases
        self.categories = categories
        self.products = products

        self.rename_columns()
        self.drop_unnecessary_columns()

    def rename_columns(self):
        # Rename columns for better readability
        self.purchases.rename(
            columns={"ASIN/ISBN (Product Code)": "product_code"}, inplace=True
        )
        self.products.rename(columns={"asin": "product_code"}, inplace=True)

    def drop_unnecessary_columns(self):
        # Drop unnecessary columns
        self.purchases.drop(columns=["product_code", "Survey ResponseID"], inplace=True)


    def output(self):
        # Output the processed data
        self.purchases["Order Date"] = pd.to_datetime(self.purchases["Order Date"])
        return self.purchases.set_index("Order Date")
