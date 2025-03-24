import numpy as np
import cvxpy as cp

def trend_optimizer(historical_values, historical_emissions, old_weights, end):
    """
    Uses linear regression on historical data to forecast the next-day stock values
    and emissions for each asset, and then optimizes the portfolio weights.

    Parameters:
      historical_values: 2D numpy array of shape (m, n), where m = number of past days,
                         and n = number of assets.
      historical_emissions: 2D numpy array of shape (m, n) for the emissions data.
      old_weights: 1D numpy array of length n representing current portfolio weights.

    Returns:
      The optimized weights as a numpy array.
    """
    m, n = historical_values.shape  # m days, n assets
    t = np.arange(m)  # time index for the past m days

    # Initialize forecast arrays
    forecast_values = np.zeros(n)
    forecast_emissions = np.zeros(n)

    # Compute linear trend (slope and intercept) for each asset and forecast the next day's value.
    for i in range(n):
        # Forecast stock value for asset i:
        slope_val, intercept_val = np.polyfit(t, historical_values[:, i], 1)
        forecast_values[i] = slope_val * end + intercept_val  # forecast for day m (next day)
        
        # Forecast emissions for asset i:
        slope_em, intercept_em = np.polyfit(t, historical_emissions[:, i], 1)
        forecast_emissions[i] = slope_em * end + intercept_em

    # Now, forecast_values and forecast_emissions are treated as fixed parameters
    # in the optimization model.

    # Variable to optimize: new portfolio weights
    weights = cp.Variable(n)

    # Set weightings (alpha for emissions, beta for returns/values)
    # Adjust these coefficients as needed for your trade-off.
    alpha = 1   # Weighting for emissions penalty
    beta = 0.02    # Weighting for value benefit

    # Define the objective:
    # Here we aim to minimize forecasted emissions while maximizing forecasted values.
    # Since CVXPY minimizes, we subtract the benefit from the cost.
    objective = cp.Minimize(alpha * cp.sum(cp.multiply(forecast_emissions, weights)) -
                            beta * cp.sum(cp.multiply(forecast_values, weights)))

    # For reference, compute the current portfolio's forecast metrics.
    old_forecast_emissions = np.sum(forecast_emissions * old_weights)
    old_forecast_values = np.sum(forecast_values * old_weights)

    # Define constraints:
    constraints = [
        # For example, require that the new portfolio has lower total emissions than the old one
        cp.sum(cp.multiply(forecast_emissions, weights)) <= old_forecast_emissions,
        # And that it retains at least the same level of forecasted value
        cp.sum(cp.multiply(forecast_values, weights)) >= old_forecast_values,
        cp.sum(weights) == 1,          # Portfolio weights must sum to 1
        weights >= 0,                  # No short selling (non-negative weights)
        #cp.sum(cp.abs(weights - old_weights)) <= 0.5,  # Total change in weights allowed
        cp.abs(weights - old_weights) <= 0.05          # Maximum change per asset
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return weights.value