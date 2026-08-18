import unittest
import pandas as pd
from src.data_cleaner import DataCleaner
from src.analyzer import Analyzer

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Transaction ID': [1, 2, 2, 3],
            'Total Amount': [100.0, 200.0, 200.0, None],
            'Category': ['A', 'B', 'B', 'A']
        })

    def test_cleaning(self):
        cleaned = DataCleaner.clean_data(self.df)
        self.assertEqual(cleaned.duplicated().sum(), 0)

    def test_stats(self):
        stats = Analyzer.get_summary_stats(self.df)
        self.assertIsNotNone(stats)

if __name__ == '__main__':
    unittest.main()