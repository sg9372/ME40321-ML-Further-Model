import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

def train_xgboost_model(current_values: np.ndarray, future_values: np.ndarray,model=None):
    # Calculate the average future values for each stock
    future_averages = np.mean(future_values, axis=0)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(current_values.T, future_averages, test_size=0.2, random_state=42)
    
    '''
    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predict the average future values for the test set
    predictions = model.predict(X_test)
    '''

    # Prepare the data for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set the parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Train the XGBoost model
    if model is None: 
        print("NEW MODEL")
        model = xgb.train(params, dtrain, num_boost_round=100)
    else:
        model = xgb.train(params, dtrain, num_boost_round=100, xgb_model=model)

    # Predict the average future values for the test set
    predictions = model.predict(dtest)
    
    #if printResults:
    predictions_error = np.sum(np.abs(predictions - y_test))/(len(y_test)*np.mean(y_test))
    averages_error = np.sum(np.abs(np.mean(current_values, axis=0) - future_averages))/(len(future_averages)*np.mean(future_averages))
    print(f'Error in predictions: {predictions_error}')
    print(f'Error in averages: {averages_error}')
    #print("\nPredictions",predictions)
    #print("\nY-Test:",y_test)
    return model, predictions