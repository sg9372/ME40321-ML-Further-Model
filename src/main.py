import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from src.extract_sheet_names import get_sheet_names
from src.extract_sheet_data import extract_sheet_data
from src.format_sector_weights import determine_sector_weights
from src.optimizer import average_optimizer
from src.extract_values import extract_values
from src.format_weights import format_weights
from src.get_sector_indicies import get_sector_indicies
from src.write_df import write_df
from src.ML_Models.xgboost_daily_analysis import xgboost_daily_analysis, justTest

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
   
    # Obtain sheet names and date ranges
    all_sheet_names, all_dates = get_sheet_names(file)

    sheet_names = []
    dates = []
    for i in range(len(all_dates)):
        # Skip dates before 01/09/2022
        if datetime.strptime(all_dates[i], '%d/%m/%Y') > datetime.strptime('01/09/2022', '%d/%m/%Y'):
            dates.append(all_dates[i])
            sheet_names.append(all_sheet_names[i])

    # Create a DataFrame with the top row of company tickers
    all_companies_df = pd.read_excel(file, sheet_name='ftse100_closing_prices')
    all_companies = all_companies_df.columns.tolist()[1:]
    
    optimized_weights_df = pd.DataFrame(columns=['Date'] + all_companies)
    old_weights_df = pd.DataFrame(columns=['Date'] + all_companies)

    old_sectors_df = pd.DataFrame(columns=['Date'] + all_sectors)
    optimized_sectors_df = pd.DataFrame(columns=['Date'] + all_sectors)

    # Iterate through each date range
    for i in range(len(dates)-1): # len(dates)-1
        start_date = datetime.strptime(dates[i], '%d/%m/%Y') + pd.DateOffset(days=1)
        start_date = start_date.strftime('%d/%m/%Y')
        end_date = dates[i+1]
        sheet_name = sheet_names[i]
        print(start_date)

        # Extract sheet data
        companies, weights, emissions, company_sectors = extract_sheet_data(file, sheet_name)

        # Extract values data
        values, dates_range = extract_values(file, companies, start_date, end_date)
        
        # Clean data
        if np.isnan(values).any():
            emissions = np.nan_to_num(emissions)
            weights = np.nan_to_num(weights)

        # Predict future average values
        predicted_average_values = []
        for i in range(len(companies)):
            # Remove NaN values from the column NEED TO MAKE THIS SO IT REMOVES
            values[:, i] = np.nan_to_num(values[:, i])
            if i%10==0:
                print("Company", i)
                
            # Ensure correct window_size
            if len(values[:,i])<640:
                window_size=len(values[:,i])
            else:
                window_size=640

            # Train the model
            model, _ = xgboost_daily_analysis(values[:,i], window_size, justTrain=True)
            
            # Predict
            future_predictions = justTest(model, values[-window_size:,i], 60)
            
            # Calculate the mean of future predictions
            predicted_average_values.append(np.mean(future_predictions))

        # Get sector indicies
        sector_indices = get_sector_indicies(all_sectors, company_sectors)

        # Current values
        current_values = values[-1, :]

        # Calculate optimal weights
        optimized_weights = average_optimizer(predicted_average_values, emissions, weights, current_values, sector_indices)
        
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