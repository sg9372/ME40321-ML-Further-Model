import unittest
import numpy as np
import xgboost as xgb
import pandas as pd 

from _Archive.train_xgboost_model import train_xgboost_model
from src.extract_values import extract_values
from src.extract_sheet_data import extract_sheet_data
from src.extract_sheet_names import get_sheet_names

class TestTrainXGBoostModel(unittest.TestCase):

    def setUp(self):
        # Set up some example data
        self.current_values = np.random.rand(100, 10)  # Example current period values (100 samples, 10 companies)
        self.future_values = np.random.rand(100, 10)  # Example future period values (100 samples, 10 companies)

    def test_model_training(self):
        # Test if the model is trained without errors
        model, predictions = train_xgboost_model(self.current_values, self.future_values)
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)

    def test_predictions_shape(self):
        # Test if the predictions have the correct shape
        model, predictions = train_xgboost_model(self.current_values, self.future_values)
        self.assertEqual(predictions.shape, (2,))  # 20% of 100 is 20

    def test_predictions_type(self):
        # Test if the predictions are of type numpy array
        model, predictions = train_xgboost_model(self.current_values, self.future_values)
        self.assertIsInstance(predictions, np.ndarray)

    def test_model_type(self):
        # Test if the model is of type XGBRegressor
        model, predictions = train_xgboost_model(self.current_values, self.future_values)
        self.assertIsInstance(model, xgb.Booster)
    '''
    def  test_values(self):
        # Test if the model predictions are close to the average future values
        sheet_names, dates = get_sheet_names("Monthly FTSE Data - New.xlsx")
        model = None
        n=20
        for i in range(n): 
            sheet_name = sheet_names[i]
            end_date = dates[i+1]
            companies, _, _ = extract_sheet_data("Monthly FTSE Data - New.xlsx", sheet_name)
            curr_values, fut_values, _ = extract_values("Monthly FTSE Data - New.xlsx", companies, end_date, 20)
            if i < 3:
                model, _ = train_xgboost_model(curr_values, fut_values, model=model)
            else:
                _,_ = train_xgboost_model(curr_values, fut_values, model=model)
            print(i)
            
        fut_averages = np.mean(fut_values, axis=0)'
        '''