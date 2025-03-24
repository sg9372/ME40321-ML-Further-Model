import pandas as pd
from datetime import datetime

def update_df(old_df, new_df):
    # Append new_df to old_df
    updated_df = pd.concat([old_df, new_df], ignore_index=True)
    return updated_df