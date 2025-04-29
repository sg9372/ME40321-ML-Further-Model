import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from src.extract_sheet_names import get_sheet_names
from src.extract_sheet_data import extract_sheet_data
from src.format_sector_weights import determine_sector_weights
from src.optimizer import average_optimizer
from src.extract_values import extract_values, extract_all_values
from src.format_weights import format_weights
from src.get_sector_indicies import get_sector_indicies
from src.write_df import write_df
from src.ML_Models.xgboost_daily_analysis import xgboost_daily_analysis, justTest, xgboost_add_training, new_model

def main():    
    # File name
    file = "Monthly FTSE Data - New.xlsx"

    # Sectors list
    all_sectors = [
    "Finance and Insurance",
    "Manufacturing",
    "Mining, Quarrying, and Oil and Gas Extraction",
    "Wholesale Trade",
    "Information",
    "Utilities",
    "Real Estate and Rental and Leasing",
    "Transportation and Warehousing",
    "Professional, Scientific, and Technical Services",
    "Construction",
    "Accommodation and Food Services",
    "Arts, Entertainment, and Recreation",
    "Administrative and Support and Waste Management and Remediation Services",
    "Retail Trade"
]
    # Define model start date
    optimization_start_date = '01/09/2022'

    # Obtain sheet names and date ranges
    all_sheet_names, all_dates = get_sheet_names(file)

    sheet_names = []
    dates = []
    for i in range(len(all_dates)):
        # Skip dates before start date
        if datetime.strptime(all_dates[i], '%d/%m/%Y') > datetime.strptime(optimization_start_date, '%d/%m/%Y'):
            dates.append(all_dates[i])
            sheet_names.append(all_sheet_names[i])

    # Create a DataFrame with the top row of company tickers
    all_companies_df = pd.read_excel(file, sheet_name='ftse100_closing_prices')
    all_companies = all_companies_df.columns.tolist()[1:]
    
    optimized_weights_df = pd.DataFrame(columns=['Date'] + all_companies)
    old_weights_df = pd.DataFrame(columns=['Date'] + all_companies)

    old_sectors_df = pd.DataFrame(columns=['Date'] + all_sectors)
    optimized_sectors_df = pd.DataFrame(columns=['Date'] + all_sectors)

    # Define ideal window size
    ideal_window_size = 640

    # Initial training for each company's model
    model_dict = dict.fromkeys(all_companies)
    all_values = extract_all_values(file, optimization_start_date)
    print("Training models on data up to:", optimization_start_date)
    for i in range(len(all_companies)):
        # Clean data
        values = all_values[~np.isnan(all_values[:,i]),i]
        if i%10==0:
            print("Training company:", i)
        if len(values)!=0:    
            # Create new model
            model, window_size = new_model(values, ideal_window_size)
            model_dict[all_companies[i]] = {"model" : model, "window_size" : window_size} 

    # Iterate through each date range
    print(len(dates))
    for i in range(1, len(dates)-1): # len(dates)-1
        start_date = dates[i-1]
        curr_date = dates[i]
        end_date = dates[i+1]
        sheet_name = sheet_names[i]
        print(start_date)

        # Extract sheet data
        companies, weights, emissions, company_sectors = extract_sheet_data(file, sheet_name)

        # Extract values data
        values, dates_range, end_date_values = extract_values(file, companies, start_date, curr_date, end_date, ideal_window_size)
        
        # Clean data
        if np.isnan(values).any():
            emissions = np.nan_to_num(emissions)
            weights = np.nan_to_num(weights)

        # Predict future average values
        predicted_values = []
        for j in range(len(companies)):
            if j%10==0:
                print("Predicting company:", j)

            # Remove NaN values from the column
            new_values = values[~np.isnan(values[:,j]),j]

            # Get model
            if model_dict[companies[j]]==None or model_dict[companies[j]]['window_size']<640: # Uncomplete model (less than 640 days of data)
                # Make new model
                model, window_size = new_model(new_values, len(new_values))
                model_dict[companies[j]] = {"model" : model, "window_size" : window_size} 
            else:                                           # Complete model (more than 640 days of data)
                # Add training to model and update model
                model_dict[companies[j]]['model'] = xgboost_add_training(model_dict[companies[j]]['model'], new_values, ideal_window_size)
            
            # Predict
            model = model_dict[companies[j]]['model']
            window_size = model_dict[companies[j]]['window_size']
            days_ahead = 13
            future_predictions = justTest(model, new_values[-window_size:], days_ahead)
            
            # Calculate the mean of future predictions
            predicted_values.append(future_predictions[-1])

        # Get sector indicies
        sector_indices = get_sector_indicies(all_sectors, company_sectors)

        # Current values
        current_values = values[-1, :]

        # Calculate optimal weights
        if i==1:    
            optimized_weights = average_optimizer(predicted_values, emissions, weights, current_values, sector_indices)
        else:
            optimized_weights = average_optimizer(predicted_values, emissions, weights, current_values, sector_indices, end_of_last_sector_portfolio_value)
        
        end_of_last_sector_portfolio_value = np.sum(end_date_values * optimized_weights)


        # Put new weights in "all company list" format
        formatted_optimized_weights_df = format_weights(all_companies, companies, optimized_weights, dates_range)
        formatted_old_weights_df = format_weights(all_companies, companies, weights, dates_range)
        # Put new sectors into correct format
        formatted_optimized_sectors_df = determine_sector_weights(all_sectors, company_sectors, optimized_weights, dates_range)
        formatted_old_sectors_df = determine_sector_weights(all_sectors, company_sectors, weights, dates_range)
        
        # Append new weights to new_weights_df
        optimized_weights_df = pd.concat([optimized_weights_df, formatted_optimized_weights_df], ignore_index=True)
        old_weights_df = pd.concat([old_weights_df, formatted_old_weights_df], ignore_index=True)
        #Append to sectors dfs.
        optimized_sectors_df = pd.concat([optimized_sectors_df, formatted_optimized_sectors_df], ignore_index=True)
        old_sectors_df = pd.concat([old_sectors_df, formatted_old_sectors_df], ignore_index=True)

        current_portfolio_value = np.sum(np.multiply(current_values, weights))

    # Write df to new sheet
    write_df(optimized_weights_df, old_weights_df, optimized_sectors_df, old_sectors_df)
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 1:                                  
        print("Wrong ammount of input arguments, example usage:")
        print("'python src/main.py'")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        sys.exit(1)