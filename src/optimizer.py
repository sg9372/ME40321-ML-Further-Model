import cvxpy as cp
import numpy as np

def average_optimizer(predicted_values, emissions, old_weights, current_values, sector_indices):
    n = len(old_weights)

    # Variable to optimize
    weights = cp.Variable(n)

    # Define objective function to minimize change between weights and old_weights
    # and maximize future value (predicted value multiplied by weights)
    predicted_portfolio_value = cp.sum(cp.multiply(predicted_values, weights))
    current_portfolio_value = cp.sum(cp.multiply(current_values, weights))

    objective = cp.Maximize(predicted_portfolio_value-current_portfolio_value)

    #change_in_weights = cp.norm(weights - old_weights, 1)
    #objective = cp.Minimize(change_in_weights - predicted_values)

    # Calculate current emissions and value
    old_emissions = np.sum(emissions * old_weights)
    old_predicted_valuess = np.sum(current_values * old_weights)

    # Sector limit
    sector_limit = 0.025

    # Define constraints
    constraints = [
        cp.sum(cp.multiply(emissions, weights)) <= old_emissions*0.8,   # Emissions cap
        cp.sum(weights) == 1,
        weights >= 0,
        cp.sum(cp.multiply(current_values, weights)) == cp.sum(cp.multiply(current_values, old_weights)), # Means we can't magic in more money
        cp.sum(cp.abs(weights - old_weights)) <= 0.15,  # Total change in weights
    ]

    for indices in sector_indices:
        if indices:
            constraints.append(cp.sum(cp.abs(weights[indices] - old_weights[indices])) <= sector_limit) 

    # Solve
    prob = cp.Problem(objective, constraints)
    
    prob.solve(verbose=False)

    # Output Weights
    return weights.value