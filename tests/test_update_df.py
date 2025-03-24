import unittest
import pandas as pd
from src.update_df import update_df

class test_add_first_line(unittest.TestCase):
    # Test number of companies
    def test_first_pass_shape(self):
        # Test if the function correctly appends first line to df.
        optimized_weights_df = pd.DataFrame(columns=['Date'] + ['Company1', 'Company2', 'Company3', 'Company4', 'Company5'])
        first_row_df = pd.DataFrame([['2021-01-01', 0.1, 0.2, 0.3, 0.4, 0.5]], columns=['Date', 'Company1', 'Company2', 'Company3', 'Company4', 'Company5'])
        
        result = update_df(optimized_weights_df, first_row_df)
        self.assertEqual(result.shape, (1, 6))

    def test_append_shape(self):
        # Test if the function correctly appends to df.
        optimized_weights_df = pd.DataFrame(columns=['Date'] + ['Company1', 'Company2', 'Company3', 'Company4', 'Company5'])
        first_row_df = pd.DataFrame([['2021-01-01', 0.1, 0.2, 0.3, 0.4, 0.5]], columns=['Date', 'Company1', 'Company2', 'Company3', 'Company4', 'Company5'])
        optimized_weights_df = update_df(optimized_weights_df, first_row_df)


        result = update_df(optimized_weights_df, first_row_df)
        self.assertEqual(result.shape, (2, 6))