import numpy as np
import xgboost as xgb
import time

from sklearn.model_selection import train_test_split
from src.extract_values import extract_values, extract_all_values
from src.extract_sheet_data import extract_sheet_data
from src.extract_sheet_names import get_sheet_names
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.ML_Models.xgboost_daily_analysis import xgboost_daily_analysis, plot_data, justTest, xgboost_add_training, new_model
import pandas as pd


def test_real_data():
    abs_max_error = 0
    # Test if the model can be trained with real data
    companies = ['DWL.L', 'STAN.L', 'HBR.L']      # HBR.L, JE.L, STAN.L
    window_sizes = [0,2,5,10,20,40,80,160,320,640] # [2,5,10,20,40,80,160,320,640]
    window_sizes.reverse()
    all_data = extract_all_values('Monthly FTSE Data - New.xlsx', '01/12/2023')
    days = 20    
    step_size = 10

    incomplete_companies = []
    
    start_company = 0
    for i in range(start_company, 61): # len(all_data[0,:])
    #for i in range(30):
        print("Company", i)
        data = all_data[:, i]
        if len(data) == 946 and  not np.isnan(data).any():
            results = { 'window_size': [], 'Error': []}
            
            for ideal_window_size in window_sizes:
                predErrors = []
                if ideal_window_size != 0:                
                    print("Window Size", ideal_window_size)
                    model, _ = xgboost_daily_analysis(data[:max(window_sizes)+2], ideal_window_size, justTrain=True)

                    for date in range(max(window_sizes)+2+step_size, 946-days, step_size): # 946-60 # Must be max + 2, max + 1 gives sufficient space for biggest window then 1 more for target value
                        new_data = data[date-step_size-ideal_window_size:date]
                        # Train the model with the new data
                        model = xgboost_add_training(model, new_data, ideal_window_size)

                        y_pred = justTest(model, data[date-ideal_window_size:date], days)

                        
                        #predErrors.append((data[date+days] - y_pred[-1][0])/data[date+days])
                        #if abs((data[date+days] - y_pred[-1][0]) > abs_max_error):
                        #        print("Error", data[date+days] - y_pred[-1][0])
                        #        print("Input data", data[date-ideal_window_size:date])
                        #plot_data(data[date:date+days], y_pred, data[:date])
                        #        abs_max_error = abs(data[date+days] - y_pred[-1][0])

                
                else:
                    for date in range(max(window_sizes)+2, 946-days, step_size): # 946-60 # Must be max + 2, max + 1 gives sufficient space for biggest window then 1 more for target value
                        predError = data[date+days] - data[date]
                        predErrors.append(predError/data[date+days])

                results['window_size'] += [ideal_window_size] * len(predErrors)
                results['Error'] += predErrors
            
            
            if i == start_company:

                final_df = pd.DataFrame(results)
            else:
                results = pd.DataFrame(results['Error'])
                final_df = pd.concat([final_df, results], axis=1)
        else:
            incomplete_companies.append(i)
            print("Appended company", i)
        
        if i % 5 == 0:
            print(incomplete_companies)
            final_df.to_excel('xgboost_test_results_tester_GPTParams.xlsx', index=False)
            # Add incomplete companies to a new sheet in the Excel file
            with pd.ExcelWriter('xgboost_test_results_tester_GPTParams.xlsx', mode='a', engine='openpyxl') as writer:
                pd.DataFrame({'Incomplete Companies': incomplete_companies}).to_excel(writer, sheet_name='Incomplete Companies', index=False)
            

test_real_data()