import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from features.featureengineering import FeatureEngineering

import pandas as pd


class TestFeatureEngineering(unittest.TestCase):
    """TestFeatureEngineering is a test case for the FeatureEngineering class.
    It tests for the functionalities in the FeatureEngineering class.
    
    

    Args:
        unittest (_type_): _description_
    """

    def __init__(self, methodToRun="runTest"):
        super().__init__(methodToRun)

    def runTest(self):
        """Test to check the functionality of FeatureEngineering class in istrain=True"""
        # Sample data for testing
        data = pd.DataFrame(
            {
                "total_sales": [100, 200, 300, 400, 500],
                "Shipping Address State": ["CA", "NY", "TX", "CA", "NY"],
                "Category": ["A", "B", "C", "D", "E"],
                "Order Date": pd.to_datetime(
                    [
                        "2018-01-01",
                        "2019-01-01",
                        "2020-01-01",
                        "2021-01-02",
                        "2022-01-03",
                    ]
                ),
            }
        ).set_index("Order Date")
        holidays_2021 = pd.DataFrame(
            {
                "holiday": [False, True, False, True, False],
                "Date": pd.to_datetime(
                    [
                        "2018-01-01",
                        "2019-01-01",
                        "2020-01-01",
                        "2021-01-02",
                        "2022-01-03",
                    ]
                ),
            }
        ).set_index("Date")
        events = pd.DataFrame(
            {
                "Amazon Events": ["event1", "event2"],
                "Date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            }
        ).set_index("Date")
        holidays = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "is_holiday": [True, False],
            }
        )
        # Initialize FeatureEngineering class
        fe = FeatureEngineering(data, holidays_2021, events, holidays=holidays)

        # Test the add_holidays method
        fe.add_holidays()

        # Check if the holidays were added correctly
        self.assertIn("fedral_holiday", fe.output().columns)


if __name__ == "__main__":
    unittest.main()
