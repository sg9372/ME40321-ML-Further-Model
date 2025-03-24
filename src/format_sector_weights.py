import numpy as np
import pandas as pd

# Format weights correctly
def determine_sector_weights(all_sectors: list, company_sectors: list, weights: np.ndarray, dates_range):
    sector_index_map = {sector: idx for idx, sector in enumerate(all_sectors)}
    sector_weights = [0.0] * len(all_sectors)
    for i in range(len(weights)):
        sector_weights[sector_index_map[company_sectors[i]]] += weights[i]
    
    # Create a DataFrame with dates as the LHS column and sector weights as the weights in every single row
    formatted_weights_df = pd.DataFrame([sector_weights] * len(dates_range), columns=all_sectors)
    formatted_weights_df.insert(0, 'Date', dates_range)

    return formatted_weights_df