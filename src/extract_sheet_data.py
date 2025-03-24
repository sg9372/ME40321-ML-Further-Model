import pandas as pd
import numpy as np

def extract_sheet_data(file, sheet_name):
    # Extract companies, weights and emissions data from sheet
    df = pd.read_excel(file, sheet_name=sheet_name)
    companies = df['Constituent RIC'].dropna().tolist()  # Remove NaN values and convert to list
    
    # Process company names
    for i in range(len(companies)):
        if '^' in companies[i]:
            companies[i] = companies[i].split('^')[0]

    weights = df['Weight percent'].to_numpy()
    weights = weights[~np.isnan(weights)]
    weights = weights / 100
    
    emissions = df['Emissions'].to_numpy()
    emissions = emissions[~np.isnan(emissions)]

    sectors = df['NAICS Sector Name'].dropna().tolist()
    
    return companies, weights, emissions, sectors
