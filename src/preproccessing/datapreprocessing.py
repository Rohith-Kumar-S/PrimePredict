import pandas as pd


class DataPreprocessor:
    def __init__(self, purchases, products=None, categories=None):
        self.purchases = purchases
        self.products = products
        self.categories = categories
        if "Shipping Address State" in purchases.columns:
            self.rename_columns()
            self.add_missing_data()
            self.remove_null_values()
        self.set_index()

    def rename_columns(self):
        self.purchases.rename(
            columns={"ASIN/ISBN (Product Code)": "product_code"}, inplace=True
        )
        self.products.rename(columns={"asin": "product_code"}, inplace=True)

    def set_index(self):
        self.purchases["Order Date"] = pd.to_datetime(self.purchases["Order Date"])
        self.purchases.set_index("Order Date", inplace=True)
        self.purchases = self.purchases[self.purchases.index.year < 2023]

    def add_missing_data(self):
        self.products = pd.merge(
            self.products,
            self.categories,
            how="inner",
            left_on="category_id",
            right_on="id",
        ).drop("id", axis=1)
        product_codes_category_null = self.purchases[
            self.purchases["Category"].isnull()
        ]["product_code"]

        data_replacements = self.products[
            self.products["product_code"].isin(product_codes_category_null.unique())
        ][["product_code", "title", "category_name"]]

        self.purchases = pd.merge(
            self.purchases, data_replacements, on="product_code", how="left"
        )

        self.purchases.loc[
            (self.purchases["Category"].isnull())
            & (self.purchases["category_name"].notnull()),
            "Category",
        ] = self.purchases.loc[
            (self.purchases["Category"].isnull())
            & (self.purchases["category_name"].notnull()),
            "category_name",
        ]

        self.purchases.loc[
            (self.purchases["Title"].isnull()) & (self.purchases["title"].notnull()),
            "Title",
        ] = self.purchases.loc[
            (self.purchases["Title"].isnull()) & (self.purchases["title"].notnull()),
            "title",
        ]

        self.purchases.drop(["title", "category_name"], axis=1, inplace=True)

    def remove_null_values(self):
        # Drop unnecessary columns
        self.purchases.dropna(
            axis=0, subset=["Shipping Address State", "Category"], inplace=True
        )

    def output(self):
        # Output the processed data
        return self.purchases
