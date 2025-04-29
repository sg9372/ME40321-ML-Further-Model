import cvxpy as cp
import numpy as np

def average_optimizer(predicted_values, emissions, ftse100_weights, current_values, sector_indices, end_of_last_sector_portfolio_value=None):
    n = len(ftse100_weights)

    # Variable to optimize
    weights = cp.Variable(n)

    # Calculate current emissions and value
    ftse100_emissions = np.sum(emissions * ftse100_weights)

    # Sector limit
    sector_limit = 0.025

    if end_of_last_sector_portfolio_value is not None:
        portfolio_value = end_of_last_sector_portfolio_value
    else:
        portfolio_value = np.sum(current_values * ftse100_weights)

    print("Portfolio value", portfolio_value)

    predicted_portfolio_value = cp.sum(cp.multiply(predicted_values, weights))

    #objective = cp.Maximize(predicted_portfolio_value-portfolio_value)
    change_in_weights = cp.norm(weights - ftse100_weights, 1)
    objective = cp.Minimize(change_in_weights)

    # Define constraints
    constraints = [
        cp.sum(cp.multiply(emissions, weights)) <= ftse100_emissions*0.8,   # Emissions cap
        cp.sum(weights) == 1,
        weights >= 0,
        cp.sum(cp.multiply(current_values, weights)) == portfolio_value, # Means we can't magic in more money
        cp.sum(cp.abs(weights - ftse100_weights)) <= 0.2,  # Total change in weights
        #cp.max(cp.abs(weights - ftse100_weights)) <= 0.01,
    ]

    for indices in sector_indices:
        if indices:
            constraints.append(cp.sum(cp.abs(weights[indices] - ftse100_weights[indices])) <= sector_limit) 

    # Solve
    prob = cp.Problem(objective, constraints)
    
    prob.solve(verbose=False)

    # Output Weights
    return weights.value