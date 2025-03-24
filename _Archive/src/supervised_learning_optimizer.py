import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def supervised_learning_optimizer(values: np.ndarray, emissions: np.ndarray, old_weights: np.ndarray) -> np.ndarray:
    n = len(old_weights)

    # Train a regression model to predict the portfolio's value based on the weights
    model = LinearRegression()
    model.fit(values.T, old_weights)
    feature_importance = model.coef_

    # Sort companies by feature importance (least impact on value first)
    sorted_indices = np.argsort(np.abs(feature_importance))

    # Objective: Minimize the L1-norm difference between new weights and old_weights
    def objective(weights):
        return np.sum(np.abs(weights - old_weights))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Weights must sum to 1
        {'type': 'ineq', 'fun': lambda weights: np.sum(emissions * weights) - np.sum(emissions * old_weights) * 0.95},  # Reduce emissions by 5%
        {'type': 'ineq', 'fun': lambda weights: np.sum(values @ weights) - np.sum(values @ old_weights)}  # Maintain value
    ]

    # Bounds for weights (non-negative)
    bounds = [(0, None) for _ in range(n)]

    # Initial guess is the old weights
    initial_guess = old_weights

    # Run the optimizer
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        print("Optimization failed: " + result.message)

    optimized_weights = result.x
    return optimized_weights