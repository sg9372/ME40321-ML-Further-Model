import pandas as pd
import numpy as np

# Extract all values before curr_date while giving dates for next 60 rows.
def extract_values(file, companies, curr_Date, end_Date):
    # Read data
    df = pd.read_excel(file, sheet_name='ftse100_closing_prices') 
    
    # Convert from UK to US date format (DD/MM/YYYY to MM/DD/YYYY)
    US_curr_Date = pd.to_datetime(curr_Date, dayfirst=True).strftime('%m/%d/%Y')
    US_end_Date = pd.to_datetime(end_Date, dayfirst=True).strftime('%m/%d/%Y')

    # Filter based on the Date range
    filtered_df = df[(df['Date'] >= '01/01/2020') & (df['Date'] <= US_curr_Date)].copy()
    filtered_df.interpolate(method='linear', inplace=True, limit_direction='both', axis=0)


    # Get the Dates column data for the next 60 rows after the filtered_df
    future_dates = df[(df['Date'] > US_curr_Date) & (df['Date'] <= US_end_Date)]
    
    # Calculate get data for the actual companies we want
    company_values = filtered_df[companies].to_numpy()

    return np.round(company_values, 2), future_dates['Date']