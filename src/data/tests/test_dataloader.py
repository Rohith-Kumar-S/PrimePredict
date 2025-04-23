import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data.dataloader import DataLoader


class TestDataLoader(unittest.TestCase):
    """TestDataLoader is a test case for the DataLoader class."""

    def test_data_loader(self):
        loaded_data = DataLoader()
        event_dates = loaded_data.add_events(
            [["2024-03-20", "2024-03-25"], ["2025-03-25", "2025-03-31"]]
        )
        self.assertEqual(len(event_dates), 13)
        
    


if __name__ == "__main__":
    unittest.main(verbosity=3)
