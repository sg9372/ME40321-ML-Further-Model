import pandas as pd
from datetime import datetime 

'''
def write_df(file, new_weights_df):
    # Define sheet name
    new_sheet_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with pd.ExcelWriter(new_sheet_name, engine='openpyxl', mode='a') as writer:
        new_weights_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    return None'''

def write_df(new_weights_df, old_weights_df, optimized_sectors, old_sectors):
    # Define new file name
    new_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
    
    # Write the DataFrame to the new file
    with pd.ExcelWriter(new_file_name, engine='openpyxl') as writer:
        new_weights_df.to_excel(writer, sheet_name='New Weights', index=False)
        old_weights_df.to_excel(writer, sheet_name='Old Weights', index=False)
        optimized_sectors.to_excel(writer, sheet_name='Optimized Sectors', index=False)
        old_sectors.to_excel(writer, sheet_name='Old Sectors', index=False)
    
    return None