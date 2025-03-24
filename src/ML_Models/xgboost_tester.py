import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from src.extract_values import extract_values
from src.extract_sheet_data import extract_sheet_data
from src.extract_sheet_names import get_sheet_names
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.ML_Models.xgboost_daily_analysis import xgboost_daily_analysis, plot_data, justTest
import pandas as pd

def calculate_values(y_test, y_pred):
    # Calculate prediction error
    error = np.mean(y_test) - np.mean(y_pred)
    return error

def test_real_data():
    # Test if the model can be trained with real data
    companies = ['JE.L']      # HBR.L, JE.L, STAN.L
    window_sizes = [2,5,10,20,40,80,160,320,640]
    data, _ = extract_values('Monthly FTSE Data - New.xlsx', companies, '01/09/2023', '01/12/2023')
    data = data[:,0]
    days = 60
    results_df = pd.DataFrame(columns=['date', 'window_size', 'LastQAvgError', 'MLError'])
    for date in range(max(window_sizes)+200, 851): # 946-60 # Must be max + 2, max + 1 gives sufficient space for biggest window then 1 more for target value
        print(date)
        for window_size in window_sizes:   
            y_train = data[:date]
            model, _ = xgboost_daily_analysis(y_train, window_size, justTrain=True)
            y_pred = justTest(model, data[date-window_size:date], days)
            y_test = data[date:date+days]
            predError = calculate_values(y_test, y_pred)
            plot_data(y_test, y_pred, y_train)
            lastQAvg = np.mean(y_test)-np.mean(y_train[-60:])
            results = {
                'date': date,
                'window_size': window_size,
                'LastQAvgError': lastQAvg,
                'MLError': predError,
            }
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        if date%50==0:
            results_df.to_excel('xgboost_test_results_integration2.xlsx', index=False)

test_real_data()