import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def xgboost_daily_analysis(values: np.ndarray, window_size, reg_alpha=None, reg_lambda=None,justTrain=False):
    # Create features and target variable using a rolling window of 60 days
    X = []
    y = []

    # Feature engineering
    for i in range((len(values) - window_size)):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])
    
    # Turn back into arrays
    X = np.array(X)
    y = np.array(y)

    # Split the scaled data into train and test sets
    test_size = 0.01 if justTrain else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and fit the XGBoost model
    model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='mae',
    eta=0.05,
    max_depth=5,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.7,
    gamma=0.4,
    reg_lambda=10,
    reg_alpha=0.00001,
    n_estimators=1000 # You can change this or add early stopping
)
    model.fit(X_train, y_train)

    # Predict stock prices
    y_pred = model.predict(X_test)

    return model, y_pred

def justTest(model, window_data, days):
    future_predictions = []

    window = window_data.reshape(1,-1)

    for _ in range(days):
        # Get next prediction.
        next_pred = model.predict(window)
        if abs(next_pred[0] - window[0, -1]) > 0.02 * window[0, -1]:
            if next_pred[0] < window[0, -1]:
                next_pred[0] = 0.98 * window[0, -1]
            else:
                next_pred[0] = 1.02 * window[0, -1]


        future_predictions.append(next_pred[0]) 
        window = np.append(window[0,1:], next_pred, axis=0).reshape(1, -1)

    return np.array([future_predictions]).T

def plot_data(y_test, y_pred, y_train=None):
    TrueData = np.concatenate((y_train, y_test), axis=0)
    #print("Y_Train Shape:", y_train.shape)
    #print("Y_Pred Shape:", y_pred)
    predictedData = np.concatenate((y_train, y_pred[:,0]), axis=0)
    
    dataLen = len(predictedData)
    dates = np.linspace(1, dataLen, dataLen)
    
    # Calculate RMSE, MAE, and R-squared
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Error (MAE):', mae)
    print('R-squared:', r_squared)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, TrueData, label='Actual')
    plt.plot(dates, predictedData, label='Predicted')
    plt.ylim(0, 1000)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('XGBoost Predictions')
    plt.legend()
    plt.show()

def xgboost_add_training(model, values: np.ndarray, window_size):
    # Create features and target variable using a rolling window
    X = []
    y = []

    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])

    # Turn back into arrays
    X = np.array(X)
    y = np.array(y)

    # Train the model with the new data
    model.fit(X, y, xgb_model=model.get_booster())

    return model

def new_model(values, ideal_window_size):
    if len(values)==0:
        return
    
    if len(values)<ideal_window_size+2:
        window_size=len(values)-5
    else:
        window_size=640
    
    #print("New model window size:", window_size)
    
    new_model, _ = xgboost_daily_analysis(values, window_size, justTrain=True)
    
    return new_model, window_size