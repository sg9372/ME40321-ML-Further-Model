import unittest
import numpy as np
import pandas as pd

from src.format_sector_weights import determine_sector_weights

class TestDetermineSectorWeights(unittest.TestCase):
    '''
    def test_basic_case(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = ["Tech", "Finance", "Tech", "Healthcare"]
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        expected = [0.6, 0.3, 0.1]
        result = determine_sector_weights(all_sectors, company_sectors, weights)
        np.testing.assert_almost_equal(result, expected)

    def test_empty_weights(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = []
        weights = np.array([])
        expected = [0.0, 0.0, 0.0]
        result = determine_sector_weights(all_sectors, company_sectors, weights)
        self.assertEqual(result, expected)

    def test_single_sector(self):
        all_sectors = ["Tech"]
        company_sectors = ["Tech", "Tech", "Tech"]
        weights = np.array([0.5, 0.3, 0.2])
        expected = [1.0]
        result = determine_sector_weights(all_sectors, company_sectors, weights)
        self.assertEqual(result, expected)

    def test_unmatched_sectors(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = ["Tech", "Finance"]
        weights = np.array([0.5, 0.5])
        expected = [0.5, 0.5, 0.0]
        result = determine_sector_weights(all_sectors, company_sectors, weights)
        self.assertEqual(result, expected)

    def test_zero_weights(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = ["Tech", "Finance", "Healthcare"]
        weights = np.array([0.0, 0.0, 0.0])
        expected = [0.0, 0.0, 0.0]
        result = determine_sector_weights(all_sectors, company_sectors, weights)
        self.assertEqual(result, expected)
    '''
    def test_with_dates_range(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = ["Tech", "Finance", "Tech", "Healthcare"]
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        dates_range = pd.date_range(start="2023-01-01", end="2023-01-03")
        expected_data = {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Tech": [0.6, 0.6, 0.6],
            "Finance": [0.3, 0.3, 0.3],
            "Healthcare": [0.1, 0.1, 0.1],
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = determine_sector_weights(all_sectors, company_sectors, weights, dates_range)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_empty_dates_range(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = ["Tech", "Finance", "Healthcare"]
        weights = np.array([0.4, 0.3, 0.3])
        dates_range = pd.date_range(start="2023-01-01", end="2023-01-01")
        expected_data = {
            "Date": ["2023-01-01"],
            "Tech": [0.4],
            "Finance": [0.3],
            "Healthcare": [0.3],
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = determine_sector_weights(all_sectors, company_sectors, weights, dates_range)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_no_companies_with_dates(self):
        all_sectors = ["Tech", "Finance", "Healthcare"]
        company_sectors = []
        weights = np.array([])
        dates_range = pd.date_range(start="2023-01-01", end="2023-01-03")
        expected_data = {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Tech": [0.0, 0.0, 0.0],
            "Finance": [0.0, 0.0, 0.0],
            "Healthcare": [0.0, 0.0, 0.0],
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = determine_sector_weights(all_sectors, company_sectors, weights, dates_range)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_single_date_single_sector(self):
        all_sectors = ["Tech"]
        company_sectors = ["Tech", "Tech", "Tech"]
        weights = np.array([0.5, 0.3, 0.2])
        dates_range = pd.date_range(start="2023-01-01", end="2023-01-01")
        expected_data = {
            "Date": ["2023-01-01"],
            "Tech": [1.0],
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = determine_sector_weights(all_sectors, company_sectors, weights, dates_range)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

if __name__ == "__main__":
    unittest.main()