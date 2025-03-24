import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error

def train_gpr_model(current_values: np.ndarray, future_values: np.ndarray, model=None):
    # Calculate the average future values for each stock
    future_averages = np.mean(future_values, axis=0)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(current_values.T, future_averages, test_size=0.2, random_state=42)
    
    # Set the kernel for GPR
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    
    # Train the GPR model
    if model is None:
        print("NEW MODEL")
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Predict the average future values for the test set
    predictions = model.predict(X_test) if X_test is not None else None

    predictions_error = np.sum(np.abs(predictions - y_test)) / (len(y_test) * np.mean(y_test))
    averages_error = np.sum(np.abs(np.mean(current_values, axis=0) - future_averages)) / (len(future_averages) * np.mean(future_averages))
    print(f'Error in predictions: {predictions_error}')
    print(f'Error in averages: {averages_error}')
    print("\nPredictions", predictions)
    print("\nY-Test:", y_test)
    
    # Calculate and print RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f'RMSE: {rmse}')
    
    return model, predictions