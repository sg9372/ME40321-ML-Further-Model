import unittest
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from src.extract_values import extract_values
from src.extract_sheet_data import extract_sheet_data
from src.extract_sheet_names import get_sheet_names
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.ML_Models.xgboost_daily_analysis import xgboost_daily_analysis, plot_data, justTest

class TestXGBoostDailyAnalysis(unittest.TestCase):

    def setUp(self):
        # Set up some example data
        self.data = np.random.rand(100, 1)  # Example numpy array with 100 rows and 1 column

    def test_model_training(self):
        # Test if the model is trained without errors
        model, predictions = xgboost_daily_analysis(self.data, 5)
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
    
    def test_predictions_shape(self):
        # Test if the predictions have the correct shape
        model, predictions = xgboost_daily_analysis(self.data, 1)
        self.assertEqual(predictions.shape, (20,))  # 20% of 100 is 20

    def test_predictions_type(self):
        # Test if the predictions are of type numpy array
        model, predictions = xgboost_daily_analysis(self.data, 1)
        self.assertIsInstance(predictions, np.ndarray)

    def test_model_type(self):
        # Test if the model is of type XGBRegressor
        model, predictions = xgboost_daily_analysis(self.data, 1)
        self.assertIsInstance(model, xgb.XGBRegressor)
    
    def test_metrics(self):
        # Test if the metrics are calculated correctly using a window size
        window_size = 5
        model, predictions = xgboost_daily_analysis(self.data, window_size)
        _, X_test, _, y_test = train_test_split(self.data[:-window_size], self.data[window_size:], test_size=0.2, random_state=42)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r_squared = r2_score(y_test, predictions)
        self.assertGreaterEqual(rmse, 0)
        self.assertGreaterEqual(mae, 0)
        self.assertLessEqual(r_squared, 1)
        self.assertGreaterEqual(r_squared, -5)
    
    def test_real_data(self):
        # Test if the model can be trained with real data
        companies = ['STAN.L']
        mid_date = '01/06/2022'
        data, _ = extract_values('Monthly FTSE Data - New.xlsx', companies, '01/01/2020', mid_date)
        future_data, _ = extract_values('Monthly FTSE Data - New.xlsx', companies, mid_date, '01/09/2023')
        window_size = 100
        days = 60
        print(data.shape)
        model, _ = xgboost_daily_analysis(data, window_size)
        predictions = justTest(model, data[-window_size:], days)
        plot_data(future_data[:days], predictions, y_train=data)
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
    '''   
    def test_justTest_predictions_length(self):
        # Test if justTest returns the correct number of predictions
        window_size = 5
        model, _ = xgboost_daily_analysis(self.data, window_size, justTrain=True)
        predictions = justTest(model, self.data[-window_size:], 10)
        self.assertEqual(len(predictions), 10)
    
    def test_justTest_predictions_type(self):
        # Test if justTest returns predictions of type numpy array
        window_size = 5
        model, _ = xgboost_daily_analysis(self.data, window_size, justTrain=True)
        predictions = justTest(model, self.data[-window_size:], 10)
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_justTest_predictions_values(self):
        # Test if justTest returns valid prediction values
        window_size = 5
        model, _ = xgboost_daily_analysis(self.data, window_size, justTrain=True)
        predictions = justTest(model, self.data[-window_size:], 10)
        self.assertTrue(np.all(predictions >= 0))  # Assuming stock prices are non-negative'''