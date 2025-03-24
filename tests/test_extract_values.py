import unittest
import numpy as np
import pandas as pd
from src.extract_sheet_data import extract_sheet_data
from src.extract_values import extract_values
import os

class test_extract_values(unittest.TestCase):
    def testNoError(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['STAN.L', 'JE.L'], '01/09/2022', '02/10/2022')
        self.assertIsNotNone(values)
        self.assertIsNotNone(dates)
    
    def testValuesDataType(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['STAN.L', 'JE.L'], '01/09/2022', '02/10/2022')
        self.assertIsInstance(values, np.ndarray)

    def testDatesDataType(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['STAN.L', 'JE.L'], '01/09/2022', '02/10/2022')
        self.assertIsInstance(dates, pd.Series)

    def testDatesLength(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['STAN.L', 'JE.L'], '01/09/2022', '02/10/2022')
        self.assertEqual(len(dates), 20)

    def testValuesLength(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['STAN.L','JE.L'], '01/09/2022', '02/10/2022')
        print(values.shape)
        print(len(values[:,0]))
        print(len(values[0]))
        self.assertEqual(values.shape[0], 675)
        self.assertEqual(values.shape[1], 2)
    
    def testRemoveNaNValues(self):
        values, dates = extract_values('Monthly FTSE Data - New.xlsx', ['MGGT.L'], '01/09/2022', '02/10/2022')
        # Check if there are no NaN values in the extracted values
        self.assertIsNotNone(values)


if __name__ == '__main__':
    unittest.main()