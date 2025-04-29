import unittest
import numpy as np
import pandas as pd
from src.extract_sheet_data import extract_sheet_data
from src.extract_values import extract_values

class TestExtractValues(unittest.TestCase):
    def test_extract_values_window_size(self):
        file = 'Monthly FTSE Data - New.xlsx'
        companies = ['STAN.L', 'JE.L']
        start_date = '04/01/2022'
        curr_Date = '01/02/2022'
        end_Date = '01/03/2022'
        window_size = 10

        # Dates length = 20, Values length = 31
        values, dates = extract_values(file, companies, start_date, curr_Date, end_Date, window_size)
        print("Values: ", values)
        # Assertions
        self.assertEqual(values.shape[0], 30)
        self.assertEqual(values.shape[1], 2)
        self.assertEqual(len(dates), 20)