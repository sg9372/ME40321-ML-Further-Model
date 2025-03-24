import unittest
import numpy as np
from _Archive.src.trends_extract import extract_trends

class test_extract_trends(unittest.TestCase):

    # First row of vlaues.
    def test_RowOneValues(self):
        correct = [101.13, 201.55, 305.22, 402.88, 509.33, 605.77, 712.22, 808.66, 915.11, 1011.55]
        [values,_] = extract_trends('sample_data.xlsx',1,5)
        np.testing.assert_array_equal(correct, values[0])
    
    # First row of emissions.
    def test_RowOneEmissions(self):
        correct = [4.77, 9.21, 14.78, 19.55, 24.66, 29.11, 34.44, 39.77, 45.11, 50.44]
        [_,emissions] = extract_trends('sample_data.xlsx', 1, 1)
        np.testing.assert_array_equal(correct, emissions[0])
    
    # First two rows of values.
    def test_multipleRows(self):
        correct = [
            [101.13, 201.55, 305.22, 402.88, 509.33, 605.77, 712.22, 808.66, 915.11, 1011.55],
            [103.23, 205.78, 309.87, 406.54, 512.99, 609.43, 715.88, 812.32, 918.77, 1015.22]
        ]
        [values,_] = extract_trends('sample_data.xlsx', 1, 2)
        np.testing.assert_array_equal(correct, values)

    # Random selection of emissions rows.
    def test_averageEmissions(self):
        correct = [
            [4.77, 9.21, 14.78, 19.55, 24.66, 29.11, 34.44, 39.77, 45.11, 50.44],  # Day 4
            [4.77, 9.21, 14.78, 19.55, 24.66, 29.11, 34.44, 39.77, 45.11, 50.44]   # Day 5
        ]
        [_,emissions] = extract_trends('sample_data.xlsx', 4, 5)
        np.testing.assert_array_equal(correct, emissions)