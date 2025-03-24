import pandas as pd
import numpy as np

def get_average_values(df, start_Date, end_Date):
    # Filter based on the Date range
    filtered_df = df[(df['Date'] >= start_Date) & (df['Date'] <= end_Date)]
    
    # Gather data, find mean, and round to 2dp
    value_columns = [col for col in df.columns if not col.startswith('Em_')]
    average = filtered_df[value_columns].mean().to_numpy()
    return np.round(average, 2)

def get_average_emissions(df, start_Date, end_Date):
    # Filter based on the Date range
    filtered_df = df[(df['Date'] >= start_Date) & (df['Date'] <= end_Date)]

    # Gather data, find mean, and round to 2dp
    emission_columns = [col for col in df.columns if col.startswith('Em_')]
    average = filtered_df[emission_columns].mean().to_numpy()
    return np.round(average, 2)

def extract_averages(file, start, end):
    # Read data
    df = pd.read_excel(file)    
    values = get_average_values(df, start, end)
    emissions = get_average_emissions(df, start, end)
    return [values, emissions]