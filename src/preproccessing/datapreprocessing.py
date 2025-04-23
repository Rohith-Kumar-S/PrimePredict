import pandas as pd


class DataPreprocessor:
    """
    DataPreprocessor is a class that preprocesses the data for sales forecasting.
    It handles the merging of product and category data, renames columns, and adds new features.
    """

    def __init__(
        self,
        purchases,
        products=None,
        categories=None,
        entity_to_forcast="",
        is_state=None,
    ):
        """
        Initializes the DataPreprocessor with purchases, products, categories, and entity to forecast.

        Args:
            purchases (_type_): purchases data containing sales information.
            products (_type_, optional): products data containing product information. Defaults to None.
            categories (_type_, optional): categories data containing category information. Defaults to None.
            entity_to_forcast (str, optional): _description_. Defaults to ''.
            is_state (_type_, optional): state or category to forecast. Defaults to None.
        """
        self.purchases = purchases
        self.products = products
        self.categories = categories
        self.entity_to_forcast = entity_to_forcast
        if "Shipping Address State" in purchases.columns:
            self.rename_columns()
            self.add_missing_data()
            self.remove_null_values()
            self.add_total_sales()
        if not self.entity_to_forcast == "":
            if is_state:
                self.purchases = self.purchases[
                    self.purchases["Shipping Address State"] == entity_to_forcast
                ]
            else:
                print("bleh")
                print(entity_to_forcast)
                self.purchases = self.purchases[
                    self.purchases["Category"] == entity_to_forcast
                ]
        self.set_index()

    def rename_columns(self):
        """rename_columns renames the columns in the purchases and products dataframes to standardize them."""

        self.purchases.rename(
            columns={"ASIN/ISBN (Product Code)": "product_code"}, inplace=True
        )
        self.products.rename(columns={"asin": "product_code"}, inplace=True)

    def add_total_sales(self):
        """add_total_sales calculates the total sales for each purchase and adds it to the purchases dataframe."""

        self.purchases["total_sales"] = (
            self.purchases["Purchase Price Per Unit"] * self.purchases["Quantity"]
        )

    def set_index(self):
        """set_index sets the index of the purchases dataframe to the order date and filters out data from 2023."""

        self.purchases["Order Date"] = pd.to_datetime(self.purchases["Order Date"])
        self.purchases.set_index("Order Date", inplace=True)
        self.purchases = self.purchases[self.purchases.index.year < 2023]

    def add_missing_data(self):
        """add_missing_data merges the products and categories dataframes with the purchases dataframe to fill in missing data.
        It also handles null values in the Category and Title columns.
        """
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
        """remove_null_values removes rows with null values in the Shipping Address State and Category columns."""

        # Drop unnecessary columns
        self.purchases.dropna(
            axis=0, subset=["Shipping Address State", "Category"], inplace=True
        )

    def output(self):
        """output returns the processed purchases dataframe."""
        # Output the processed data
        return self.purchases
