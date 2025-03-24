import unittest
import pandas as pd
from src.write_df import write_df
from src.update_df import update_df

class test_write_df(unittest.TestCase):
    # Test it opens and adds new sheet
    #def test_AddNewSheet(self):
    #    # Test if the function opens the file and adds a new sheet
    #    new_weights_df = pd.DataFrame(columns=['Date', 'Company A', 'Company B', 'Company C', 'Company D', 'Company E'])
    #    file = 'sample_data.xlsx'
    #    result = write_df(file, new_weights_df)
    #    self.assertEqual(result, None)

    # Test it opens and adds row of data
    def test_AddNewRow(self):
        # Test if the function opens the file and adds a new sheet
        new_weights_df = pd.DataFrame(columns=['Date', 'Company A', 'Company B', 'Company C', 'Company D', 'Company E'])
        file = 'sample_data.xlsx'
        end_date = '01/01/2020'
        new_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i in range(5):
            new_weights_df = update_df(new_weights_df, end_date,new_weights)
        result = write_df(file, new_weights_df)
        self.assertEqual(result, None)