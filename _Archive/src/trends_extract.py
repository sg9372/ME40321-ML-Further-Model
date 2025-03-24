import pandas as pd
import numpy as np

def get_values_data(df, start_day, end_day):
    # Filter based on the day range
    filtered_df = df[(df['Day'] >= start_day) & (df['Day'] <= end_day)]
    
    # Gather data, find mean, and round to 2dp
    value_columns = [col for col in df.columns if col.startswith('Value')]
    values = filtered_df[value_columns].to_numpy()
    return np.round(values, 2)

def get_emissions_data(df, start_day, end_day):
    # Filter based on the day range
    filtered_df = df[(df['Day'] >= start_day) & (df['Day'] <= end_day)]

    # Gather data, find mean, and round to 2dp
    emission_columns = [col for col in df.columns if col.startswith('Emission')]
    emissions = filtered_df[emission_columns].to_numpy()
    return np.round(emissions, 2)

def extract_trends(file, start, end):
    # Read data
    df = pd.read_excel(file)    
    
    values = get_values_data(df, start, end)
    emissions = get_emissions_data(df, start, end)
    return [values, emissions]