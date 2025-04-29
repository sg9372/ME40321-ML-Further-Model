import pandas as pd
import numpy as np

# Extract all values before curr_date while giving dates for next 60 rows.
def extract_values(file, companies, start_date, curr_Date, end_Date, window_size):
    # Read data
    df = pd.read_excel(file, sheet_name='ftse100_closing_prices') 
    
    # Convert from UK to US date format (DD/MM/YYYY to MM/DD/YYYY)
    US_start_date = pd.to_datetime(start_date, dayfirst=True).strftime('%m/%d/%Y')
    US_curr_Date = pd.to_datetime(curr_Date, dayfirst=True).strftime('%m/%d/%Y')
    US_end_Date = pd.to_datetime(end_Date, dayfirst=True).strftime('%m/%d/%Y')

    # Filter based on the Date range
    first_row = df[df['Date'] <= US_start_date].index[-1] - window_size
    print(first_row)
    df = df.reset_index()  # Reset index to ensure it's numeric
    filtered_df = df[(df.index > first_row) & (df['Date'] <= US_curr_Date)].copy()
    filtered_df.interpolate(method='linear', inplace=True, limit_area='inside', axis=0)

    # Get the Dates column data for the next 60 rows after the filtered_df
    future_dates = df[(df['Date'] > US_curr_Date) & (df['Date'] <= US_end_Date)]

    if US_end_Date in df['Date'].values:
        future_values = df[df['Date'] == US_end_Date][companies].to_numpy()
    else:
        last_date_before_end = df[df['Date'] < US_end_Date]['Date'].max()
        future_values = df[df['Date'] == last_date_before_end][companies].to_numpy()
    
    # Calculate get data for the actual companies we want
    company_values = filtered_df[companies].to_numpy()

    return np.round(company_values, 2), future_dates['Date'], np.round(future_values, 2)

def extract_all_values(file, curr_Date):
    # Read data
    df = pd.read_excel(file, sheet_name='ftse100_closing_prices') 
    
    # Convert from UK to US date format (DD/MM/YYYY to MM/DD/YYYY)
    US_curr_Date = pd.to_datetime(curr_Date, dayfirst=True).strftime('%m/%d/%Y')

    # Filter based on the Date range
    filtered_df = df[(df['Date'] >= '01/01/2020') & (df['Date'] <= US_curr_Date)].copy()
    filtered_df = filtered_df.drop('Date', axis=1)
    filtered_df.interpolate(method='linear', inplace=True, limit_direction='both', limit_area='inside', axis=0)

    
    # Calculate get data for the actual companies we want
    company_values = filtered_df.to_numpy()

    return np.round(company_values, 2)