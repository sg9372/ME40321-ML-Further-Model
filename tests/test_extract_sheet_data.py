import unittest
from src.extract_sheet_data import extract_sheet_data
import numpy as np

class test_extract_sheet_data(unittest.TestCase):
    # Test data formats
    def test_DataType(self):
        file = 'Monthly FTSE Data - New.xlsx'
        companies, weights, emissions = extract_sheet_data(file, '01_2020')
        self.assertIsInstance(companies, list)
        self.assertIsInstance(emissions, np.ndarray)
        self.assertIsInstance(weights, np.ndarray)

    # Test dateset length
    def test_Lengths(self):
        file = 'Monthly FTSE Data - New.xlsx'
        companies, weights, emissions = extract_sheet_data(file, '01_2020')
        self.assertEqual(len(companies), len(emissions))
        self.assertEqual(len(companies), len(weights))
    
    # Ensure all weightrs total 100%
    def test_WeightsTotal(self):
        file = 'Monthly FTSE Data - New.xlsx'
        _, weights, _ = extract_sheet_data(file, '01_2020')
        self.assertAlmostEqual(np.sum(weights), 1)