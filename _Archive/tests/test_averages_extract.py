import unittest
import numpy as np
from _Archive.src.averages_extract import extract_averages

class test_extract_averages(unittest.TestCase):

    def test_RowOneValues(self):
        values = [101.13, 201.55, 305.22, 402.88, 509.33, 605.77, 712.22, 808.66, 915.11, 1011.55]
        ans = extract_averages('sample_data.xlsx', 1, 1)
        np.testing.assert_array_equal(ans[0], values)

    def test_RowOneEmissions(self):
        emissions = [4.77, 9.21, 14.78, 19.55, 24.66, 29.11, 34.44, 39.77, 45.11, 50.44]
        ans = extract_averages('sample_data.xlsx', 1, 1)
        np.testing.assert_array_equal(ans[1], emissions)
    
    def test_averageValues(self):
        values = [106.37, 211.48, 315.93, 414.06, 519.38, 616.08, 722.18, 818.97, 925.42, 1021.86]
        ans = extract_averages('sample_data.xlsx', 3, 7)
        np.testing.assert_array_equal(ans[0], values)

    def test_averageEmissions(self):
        emissions = [5.03, 9.75, 15.32, 20.13, 25.33, 29.82, 35.20, 40.57, 45.95, 51.33]
        ans = extract_averages('sample_data.xlsx', 3, 7)
        np.testing.assert_array_equal(ans[1], emissions)