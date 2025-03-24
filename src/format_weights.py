import pandas as pd

# Get weights in correct format.
def format_weights(all_companies, companies, optimized_weights, dates_range):
    # Create a mapping of company names to their indices
    company_index_map = {company: idx for idx, company in enumerate(all_companies)}

    # Initialize formatted_weights
    formatted_weights = [0] * len(all_companies)

    # Populate formatted_weights
    for i in range(len(companies)):
        ind = company_index_map[companies[i]]
        formatted_weights[ind] = optimized_weights[i]
    
    # Create a DataFrame with dates as the LHS column and optimized weights as the weights in every single row
    formatted_weights_df = pd.DataFrame([formatted_weights]*len(dates_range), columns=all_companies)
    formatted_weights_df.insert(0, 'Date', dates_range)
    
    return formatted_weights_df