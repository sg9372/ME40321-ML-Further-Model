import unittest
from src.format_weights import format_weights
from src.extract_values import extract_values
from src.extract_sheet_data import extract_sheet_data

import numpy as np
import pandas as pd 

class test_format_weights(unittest.TestCase):
    def test_DataSize(self):
        # Test if the function returns the correct data size
        all_companies = ['Company A', 'Company B', 'Company C', 'Company D', 'Company E']
        companies = ['Company A', 'Company E', 'Company C']
        optimized_weights = [0.1, 0.2, 0.3]
        dates = pd.Series(["01/01/2020", "02/01/2020", "03/01/2020"])
        result = format_weights(all_companies, companies, optimized_weights, dates)
        self.assertEqual(result.shape, (len(dates), len(all_companies) + 1))
    
    def test_DataType(self):
        # Test if the function returns the correct data type
        all_companies = ['Company A', 'Company B', 'Company C', 'Company D', 'Company E']
        companies = ['Company A', 'Company E', 'Company C']
        optimized_weights = [0.1, 0.2, 0.3]
        dates = pd.Series(["01/01/2020", "02/01/2020", "03/01/2020"])
        result = format_weights(all_companies, companies, optimized_weights, dates)
        self.assertIsInstance(result, pd.DataFrame)

    def test_IntegrationTest(self):
        # Test integration with extract values and extract sheet data.
        file = "Monthly FTSE Data - New.xlsx"
        sheet_name = "ftse100_closing_prices"
        
        companies, weights, _, _ = extract_sheet_data(file, '01_2020')
        _, dates_range = extract_values(file, companies, "01/01/2020", "15/01/2020")

        all_companies_df = pd.read_excel(file, sheet_name=sheet_name)
        all_companies = all_companies_df.columns.tolist()[1:]

        result = format_weights(all_companies, companies, weights, dates_range)
        
        self.assertEqual(result.shape, (len(dates_range), len(all_companies) + 1))
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
