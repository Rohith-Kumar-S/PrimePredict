import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.forecast import SalesForecaster

class TestSalesForecaster(unittest.TestCase):
    """TestSalesForecaster is a test case for the SalesForecaster class.
    It tests if the model is loaded properly

    Args:
        unittest (_type_): 
    """
    
    def __init__(self, methodToRun="runTest"):
        super().__init__(methodToRun)
        
    def runTest(self):
        """Test to check the model loading functionality of SalesForecaster class"""
        with self.assertRaises(ValueError):
            # Test if the model is trained
            # Assuming that the model is not trained yet
            sales_forecaster = SalesForecaster(entity_name="dummy_entity")
            sales_forecaster.get_feature_importance()
            
if __name__ == "__main__":
    unittest.main()
            
        