import unittest
from src.extract_sheet_names import get_sheet_names

class test_extract_sheet_names(unittest.TestCase):
    # Test number of pages
    def test_SheetNamesPageCount(self):
        # Test if the function returns the correct sheet names
        file = 'Monthly FTSE Data - New.xlsx'
        sheet_names, _ = get_sheet_names(file)
        self.assertEqual(len(sheet_names), 61)

    def test_DatesPageCount(self):
        # Test if the function returns the correct sheet names
        file = 'Monthly FTSE Data - New.xlsx'
        _, dates = get_sheet_names(file)
        self.assertEqual(len(dates), 61)
    
    def test_DateLength(self):
        # Test if the function returns the correct sheet names
        file = 'Monthly FTSE Data - New.xlsx'
        _, dates = get_sheet_names(file)
        for date in dates:
            self.assertEqual(len(date), 10)