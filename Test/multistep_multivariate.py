# File: multistep_multivariate.py
# Task 1, 2, and 3 implementations for advanced stock prediction

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------------------------------
# Task 1: Multistep Prediction Function
#------------------------------------------------------------------------------

def create_multistep_sequences(data, price_column='Close', prediction_days=60, future_steps=5):
    """
    Create sequences for multistep prediction (predict multiple days ahead).
    
    Parameters:
    -----------
    data: pd.DataFrame
        Stock data with price columns
    price_column: str
        Column name to predict (default: 'Close')
    prediction_days: int
        Number of historical days to use as input (default: 60)
    future_steps: int
        Number of future days to predict (default: 5)
        
    Returns:
    --------
    x_data: np.array
        Input sequences (samples, time_steps, features)
    y_data: np.array
        Target sequences (samples, future_steps)
    """
    
    if price_column not in data.columns:
        raise ValueError(f"Price column '{price_column}' not found in data")
    
    price_data = data[price_column].values
    x_data, y_data = [], []
    
    # Create sequences: each x contains 'prediction_days' of historical data
    # corresponding y contains the next 'future_steps' days of prices
    for i in range(prediction_days, len(price_data) - future_steps + 1):
        # Input: historical sequence
        x_data.append(price_data[i-prediction_days:i])
        # Output: next future_steps days
        y_data.append(price_data[i:i+future_steps])
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Reshape x_data for LSTM: (samples, time_steps, features)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data

#------------------------------------------------------------------------------
# Task 2: Multivariate Prediction Function  
#------------------------------------------------------------------------------

def create_multivariate_sequences(data, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'], 
                                target_column='Close', prediction_days=60):
    """
    Create sequences for multivariate prediction (multiple input features).
    
    Parameters:
    -----------
    data: pd.DataFrame
        Stock data with multiple feature columns
    feature_columns: list
        List of column names to use as input features
    target_column: str
        Column name to predict (default: 'Close')
    prediction_days: int
        Number of historical days to use as input (default: 60)
        
    Returns:
    --------
    x_data: np.array
        Input sequences (samples, time_steps, n_features)
    y_data: np.array
        Target values (samples,)
    scalers: dict
        Dictionary of fitted scalers for each feature
    """
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Scale the features
    scalers = {}
    scaled_data = data.copy()
    
    for col in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        scalers[col] = scaler
    
    # Create sequences with multiple features
    x_data, y_data = [], []
    
    for i in range(prediction_days, len(scaled_data)):
        # Input: multiple features for 'prediction_days' historical days
        x_sequence = []
        for j in range(i-prediction_days, i):
            # Get all feature values for day j
            day_features = [scaled_data[col].iloc[j] for col in feature_columns]
            x_sequence.append(day_features)
        
        x_data.append(x_sequence)
        # Output: target column value for day i (scaled)
        y_data.append(scaled_data[target_column].iloc[i])
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data, scalers

#------------------------------------------------------------------------------
# Task 3: Combined Multivariate + Multistep Prediction Function
#------------------------------------------------------------------------------

def create_multivariate_multistep_sequences(data, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                           target_column='Close', prediction_days=60, future_steps=5):
    """
    Create sequences for multivariate, multistep prediction.
    
    Parameters:
    -----------
    data: pd.DataFrame
        Stock data with multiple feature columns
    feature_columns: list
        List of column names to use as input features
    target_column: str
        Column name to predict (default: 'Close')
    prediction_days: int
        Number of historical days to use as input (default: 60)
    future_steps: int
        Number of future days to predict (default: 5)
        
    Returns:
    --------
    x_data: np.array
        Input sequences (samples, time_steps, n_features)
    y_data: np.array
        Target sequences (samples, future_steps)
    scalers: dict
        Dictionary of fitted scalers for each feature
    """
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Scale the features
    scalers = {}
    scaled_data = data.copy()
    
    for col in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        scalers[col] = scaler
    
    # Create sequences with multiple features and multiple outputs
    x_data, y_data = [], []
    
    for i in range(prediction_days, len(scaled_data) - future_steps + 1):
        # Input: multiple features for 'prediction_days' historical days
        x_sequence = []
        for j in range(i-prediction_days, i):
            # Get all feature values for day j
            day_features = [scaled_data[col].iloc[j] for col in feature_columns]
            x_sequence.append(day_features)
        
        x_data.append(x_sequence)
        
        # Output: target column values for next 'future_steps' days (scaled)
        y_sequence = []
        for k in range(i, i + future_steps):
            y_sequence.append(scaled_data[target_column].iloc[k])
        
        y_data.append(y_sequence)
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data, scalers

#------------------------------------------------------------------------------
# Model Creation Functions for Different Tasks
#------------------------------------------------------------------------------

def create_multistep_model(sequence_length, n_features, future_steps, units=50):
    """
    Create model for multistep prediction.
    
    Parameters:
    -----------
    sequence_length: int
        Length of input sequences
    n_features: int
        Number of input features
    future_steps: int
        Number of future steps to predict
    units: int
        Number of LSTM units
        
    Returns:
    --------
    model: Sequential
        Compiled model for multistep prediction
    """
    
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    
    # Output layer - predict multiple future steps
    model.add(Dense(future_steps, activation="linear"))
    
    model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer="adam")
    
    return model

def create_multivariate_model(sequence_length, n_features, units=50):
    """
    Create model for multivariate prediction.
    
    Parameters:
    -----------
    sequence_length: int
        Length of input sequences
    n_features: int
        Number of input features
    units: int
        Number of LSTM units
        
    Returns:
    --------
    model: Sequential
        Compiled model for multivariate prediction
    """
    
    model = Sequential()
    
    # LSTM layers - can handle multiple input features
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    
    # Output layer - single prediction
    model.add(Dense(1, activation="linear"))
    
    model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer="adam")
    
    return model

def create_multivariate_multistep_model(sequence_length, n_features, future_steps, units=50):
    """
    Create model for multivariate, multistep prediction.
    
    Parameters:
    -----------
    sequence_length: int
        Length of input sequences
    n_features: int
        Number of input features
    future_steps: int
        Number of future steps to predict
    units: int
        Number of LSTM units
        
    Returns:
    --------
    model: Sequential
        Compiled model for multivariate, multistep prediction
    """
    
    model = Sequential()
    
    # LSTM layers - handle multiple features
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    
    # Output layer - predict multiple future steps
    model.add(Dense(future_steps, activation="linear"))
    
    model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer="adam")
    
    return model

#------------------------------------------------------------------------------
# Example Usage Functions
#------------------------------------------------------------------------------

def example_usage():
    """
    Example of how to use the three functions.
    """
    
    # Load your data (replace with your actual data loading)
    # data = load_your_stock_data()
    
    print("=== Task 1: Multistep Prediction ===")
    # x_multistep, y_multistep = create_multistep_sequences(data, future_steps=5)
    # model1 = create_multistep_model(60, 1, 5)
    # model1.fit(x_multistep, y_multistep, epochs=25, batch_size=32)
    
    print("=== Task 2: Multivariate Prediction ===")
    # x_multi, y_multi, scalers = create_multivariate_sequences(data)
    # model2 = create_multivariate_model(60, 5)  # 5 features
    # model2.fit(x_multi, y_multi, epochs=25, batch_size=32)
    
    print("=== Task 3: Combined Multivariate + Multistep ===")
    # x_combined, y_combined, scalers = create_multivariate_multistep_sequences(data, future_steps=5)
    # model3 = create_multivariate_multistep_model(60, 5, 5)
    # model3.fit(x_combined, y_combined, epochs=25, batch_size=32)

if __name__ == "__main__":
    example_usage() 