import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from preproccessing.datapreprocessing import DataPreprocessor
import pandas as pd


class TestDataPreprocessor(unittest.TestCase):

    def runTest(self):
        """Test to check the functionality of DataPreprocessor class for the state CA"""
        # Sample data for testing
        purchases = pd.DataFrame(
            {
                "Shipping Address State": ["CA", "NY", "TX", "CA", "NY"],
                "Category": ["A", "B", np.nan, np.nan, "E"],
                "ASIN/ISBN (Product Code)": ["P1", "P2", "P3", "P4", "P5"],
                "Purchase Price Per Unit": [10, 20, 30, 40, 50],
                "Title": ["Product 1", "Product 2", np.nan, np.nan, "Product 5"],
                "Quantity": [1, 2, 3, 4, 5],
                "Order Date": [
                    "2018-01-01",
                    "2019-01-01",
                    "2020-01-01",
                    "2021-01-02",
                    "2022-01-03",
                ],
            }
        )
        products = pd.DataFrame(
            {
                "Category": ["A", "B", "C", "D", "E"],
                "asin": ["P1", "P2", "P3", "P4", "P5"],
                "title": [
                    "Product 1",
                    "Product 2",
                    "Product 3",
                    "Product 4",
                    "Product 5",
                ],
                "category_name": ["X", "Y", "Z", "X", "Y"],
                "category_id": [1, 2, 3, 4, 5],
            }
        )
        categories = pd.DataFrame(
            {
                "Category": ["A", "B", "C", "D", "E"],
                "Subcategory": ["X", "Y", "Z", "X", "Y"],
                "id": [1, 2, 3, 4, 5],
            }
        )

        # Create an instance of DataPreprocessor
        processed_data = DataPreprocessor(
            purchases,
            products,
            categories,
            entity_to_forcast="CA",
            is_state=True,
        ).output()

        # Check if the data is preprocessed correctly
        self.assertEqual(len(processed_data), 2)  # Check number of rows
        self.assertEqual(processed_data["Title"].isnull().sum(), 0)  # No missing titles
        self.assertEqual(
            processed_data["Category"].isnull().sum(), 0
        )  # No missing categories


if __name__ == "__main__":
    unittest.main()
