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
    # range(start, stop, step)  
    #maxDepths = [3,5,7,9]
    #minChildWeights = [1,3,5]
    reg_alphas = [1e-5, 1e-2, 0.1, 1, 100]
    reg_lambdas = [0, 1, 10, 100]
    all_data = extract_all_values('Monthly FTSE Data - New.xlsx', '01/12/2023')
    days = 13    
    step_size = 25
    window_size = 640

    incomplete_companies = []
    
    start_company = 0
    for i in range(start_company, 61): # len(all_data[0,:])
    #for i in range(30):
        print("Company", i)
        data = all_data[:, i]
        if len(data) == 946 and not np.isnan(data).any():
            results = { 'reg_alpha': [], 'reg_lambda': [], 'Error': []}
            for reg_alpha in reg_alphas:
                for reg_lambda in reg_lambdas:
                    print("Reg_alpha", reg_alpha, "Reg_lambda", reg_lambda)
                    predErrors = []      
                    # Create new model
                    model, _ = xgboost_daily_analysis(data[:window_size+2], window_size, reg_alpha = reg_alpha, reg_lambda=reg_lambda,justTrain=True)

                    for date in range(window_size+2+step_size, 946-days, step_size): # 946-60 # Must be max + 2, max + 1 gives sufficient space for biggest window then 1 more for target value
                        new_data = data[date-step_size-window_size:date]
                        # Train the model with the new data
                        model = xgboost_add_training(model, new_data, window_size)
                        y_pred = justTest(model, data[date-window_size:date], days)
                        predErrors.append((data[date+days] - y_pred[-1][0])/data[date+days])

                    results['reg_alpha'] += [reg_alpha] * len(predErrors)
                    results['reg_lambda'] += [reg_lambda] * len(predErrors)
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
            final_df.to_excel('xgboost_hyperparameter_optimization_stage5.xlsx', index=False)
            # Add incomplete companies to a new sheet in the Excel file
            with pd.ExcelWriter('xgboost_hyperparameter_optimization_stage5.xlsx', mode='a', engine='openpyxl') as writer:
                pd.DataFrame({'Incomplete Companies': incomplete_companies}).to_excel(writer, sheet_name='Incomplete Companies', index=False)
            

test_real_data()