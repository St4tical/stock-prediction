# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import mplfinance as mpf
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, Bidirectional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------
# Load & Prepare Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Handle NaN values
# 3) Scale features and prepare sequences
# 4) Split into train/test sets
#------------------------------------------------------------------------------

def load_data(company='CBA.AX',           # Stock ticker symbol (default: Commonwealth Bank of Australia)
              start_date='2021-01-01',   # Start date for data collection in YYYY-MM-DD format
              end_date='2025-08-01',     # End date for data collection in YYYY-MM-DD format
              price_column='Close',      # Which price column to use for prediction (Close, Open, High, Low)
              prediction_days=60,        # Number of previous days to use for predicting next day
              split_by_date=True,        # True: split chronologically, False: split randomly
              test_size=0.2,            # Proportion of data for testing (0.2 = 20%)
              scale=True,               # Whether to normalize data to 0-1 range
              save_locally=True,        # Whether to save/load data from local files
              local_path='data',        # Directory to save/load data files
              fill_na_method='ffill',   # Method to handle missing values (ffill/bfill)
              feature_columns=['Close']):# List of columns to scale/normalize

    # REQUIREMENT 1(d): LOCAL DATA STORAGE IMPLEMENTATION
    # Create local directory if it doesn't exist and saving is enabled
    if save_locally and not os.path.exists(local_path):
        os.makedirs(local_path)
        print(f"Created directory: {local_path}")

    # Generate unique filename based on company and date range
    # This prevents conflicts when downloading different stocks or date ranges
    file_path = os.path.join(local_path, f"{company}_{start_date}_{end_date}.csv")

    # SMART CACHING MECHANISM:
    # Check if data already exists locally to avoid re-downloading
    if save_locally and os.path.exists(file_path):
        print(f"Loading existing data from: {file_path}")
        try:
            # Parse dates when loading to maintain proper datetime index
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            
            # ENHANCED CORRUPTION DETECTION AND AUTO-FIX
            # Check if data contains string values in numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data_corrupted = False
            
            print(f"Checking data integrity...")
            print(f"   Data shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            
            for col in numeric_columns:
                if col in data.columns:
                    # Check for common corruption patterns
                    sample_values = data[col].head(10).tolist()
                    print(f"   Sample values in '{col}': {sample_values[:3]}...")
                    
                    # Check if any values are strings (like 'CBA.AX')
                    try:
                        # Try to convert to numeric - this will fail if strings are present
                        pd.to_numeric(data[col].values, errors='raise')
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Corruption detected in column '{col}': {str(e)[:100]}...")
                        data_corrupted = True
                        break
                    
                    # Additional check: look for obvious string patterns
                    if data[col].dtype == 'object':
                        print(f"WARNING: Column '{col}' has object dtype - likely contains strings")
                        data_corrupted = True
                        break
            
            # If corruption detected, delete file and re-download
            if data_corrupted:
                print(f"Auto-fixing: Deleting corrupted file and downloading fresh data...")
                os.remove(file_path)
                # Re-download fresh data
                print(f"Downloading fresh data from yfinance...")
                data = yf.download(company, start=start_date, end=end_date)
                
                # FIX: Handle MultiIndex columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    print(f"Fixing MultiIndex columns...")
                    # Flatten MultiIndex columns - keep only the first level (the actual column names)
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                    print(f"   Fixed columns: {list(data.columns)}")
                
                # Verify the fresh data
                print(f"Fresh data downloaded - Shape: {data.shape}")
                print(f"   Columns: {list(data.columns)}")
                
                if save_locally:
                    data_to_save = data.reset_index()
                    data_to_save.to_csv(file_path, index=False)
                    print(f"Fresh data saved to: {file_path}")
            else:
                print(f"Data loaded successfully - no corruption detected")
                
        except Exception as e:
            print(f"ERROR: Error loading cached data: {str(e)[:100]}...")
            print(f"Auto-fixing: Downloading fresh data...")
            # If loading fails for any reason, download fresh data
            data = yf.download(company, start=start_date, end=end_date)
            
            # FIX: Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                print(f"Fixing MultiIndex columns...")
                # Flatten MultiIndex columns - keep only the first level (the actual column names)
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                print(f"   Fixed columns: {list(data.columns)}")
            
            if save_locally:
                data_to_save = data.reset_index()
                data_to_save.to_csv(file_path, index=False)
                print(f"Fresh data downloaded and saved to: {file_path}")
    else:
        print(f"Downloading fresh data for {company} from {start_date} to {end_date}")
        # Download stock data using yfinance (more reliable than pandas_datareader)
        data = yf.download(company, start=start_date, end=end_date)
        
        # FIX: Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            print(f"Fixing MultiIndex columns...")
            # Flatten MultiIndex columns - keep only the first level (the actual column names)
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            print(f"   Fixed columns: {list(data.columns)}")
        
        # Save downloaded data for future use if enabled
        if save_locally:
            # FIX: Reset index to convert MultiIndex to proper columns before saving
            data_to_save = data.reset_index()
            data_to_save.to_csv(file_path, index=False)
            print(f"Data saved to: {file_path}")

    # ADDITIONAL DATA VALIDATION AND CLEANING
    # Remove any rows that still contain string values in numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    rows_before = len(data)
    
    for col in numeric_columns:
        if col in data.columns:
            try:
                # Convert to numeric, replacing any remaining strings with NaN
                # Ensure we're working with a Series by accessing the column properly
                data[col] = pd.to_numeric(data[col].values, errors='coerce')
            except Exception as e:
                print(f"WARNING: Could not clean column '{col}': {e}")
                # If there's still an issue, try to drop the problematic column
                data = data.drop(columns=[col])
                print(f"Dropped problematic column '{col}'")
    
    # Remove rows where all price columns are NaN
    price_columns = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_columns if col in data.columns]
    if available_price_cols:
        data = data.dropna(subset=available_price_cols, how='all')
    
    rows_after = len(data)
    if rows_before != rows_after:
        print(f"🧹 Data cleaning: Removed {rows_before - rows_after} corrupted rows")
        print(f"   Final dataset: {rows_after} valid rows")

    # REQUIREMENT 1(b): NaN VALUE HANDLING - FIXED DEPRECATION WARNING
    # Handle missing values using specified method
    if fill_na_method == 'ffill':
        # Forward fill: use previous valid value to fill gaps
        data.ffill(inplace=True)  # FIXED: Updated from deprecated fillna(method='ffill')
    elif fill_na_method == 'bfill':
        # Backward fill: use next valid value to fill gaps
        data.bfill(inplace=True)  # FIXED: Updated from deprecated fillna(method='bfill')

    # REQUIREMENT 1(e): FEATURE SCALING WITH SCALER STORAGE
    # Dictionary to store fitted scalers for each feature column
    # This is crucial for inverse transformation later
    scalers = {}
    
    if scale:
        for col in feature_columns:
            if col in data.columns:  # ADDED: Safety check for column existence
                # Create individual scaler for each feature column
                # MinMaxScaler normalizes data to range [0,1]
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                # Fit scaler to column data and transform it
                # reshape(-1, 1) converts 1D array to 2D column vector (required by sklearn)
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                
                # Store fitted scaler for future inverse transformations
                # This is essential for converting predictions back to original scale
                scalers[col] = scaler
            else:
                print(f"Warning: Column '{col}' not found in data. Skipping scaling for this column.")

    # LSTM SEQUENCE PREPARATION
    # LSTM networks need sequences of data to learn temporal patterns
    # We create sliding windows of 'prediction_days' length
    x_data, y_data = [], []
    
    if price_column not in data.columns:
        raise ValueError(f"Price column '{price_column}' not found in data. Available columns: {list(data.columns)}")
    
    price_data = data[price_column].values
    
    # Create sequences: each x contains 'prediction_days' of historical data
    # corresponding y contains the next day's price to predict
    for i in range(prediction_days, len(price_data)):
        # x_data: sequence of 'prediction_days' previous prices
        x_data.append(price_data[i-prediction_days:i])
        # y_data: the price we want to predict (next day)
        y_data.append(price_data[i])
    
    # Convert lists to numpy arrays for efficiency
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # Reshape x_data for LSTM input format: (samples, time_steps, features)
    # LSTM expects 3D input: (number of sequences, sequence length, number of features)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    # REQUIREMENT 1(c): FLEXIBLE TRAIN/TEST SPLITTING
    if split_by_date:
        # CHRONOLOGICAL SPLIT (recommended for time series)
        # This respects temporal order - train on earlier data, test on later data
        
        # Calculate split index based on test_size
        if isinstance(test_size, float):
            # If test_size is float (e.g., 0.2), treat as percentage
            split_idx = int(len(x_data) * (1 - test_size))
        else:
            # If test_size is int, treat as absolute number of test samples
            split_idx = len(x_data) - test_size
            
        # Split data chronologically
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        
    else:
        # RANDOM SPLIT
        # Create array of indices and shuffle them randomly
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        
        # Calculate split point
        if isinstance(test_size, float):
            split_idx = int(len(x_data) * (1 - test_size))
        else:
            split_idx = len(x_data) - test_size
            
        # Split indices into train and test sets
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        # Use shuffled indices to create train/test sets
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

    # RETURN ALL PROCESSED DATA AND METADATA
    # data: original DataFrame with processed features
    # x_train, y_train: training sequences and targets
    # x_test, y_test: testing sequences and targets  
    # scalers: dictionary of fitted scalers for inverse transformation
    return data, x_train, y_train, x_test, y_test, scalers

#------------------------------------------------------------------------------
# Visualization Functions (Task 3 - Option C)
# 1) Candlestick chart with n-day candles
# 2) Boxplot chart for rolling windows
#------------------------------------------------------------------------------

def plot_candlestick_chart(
    df,
    n_days=1,
    title="Stock Price Candlestick Chart",
    style='charles',
    volume=True,
    save_path=None,
    figsize=(12, 8)
):
    """
    Display stock market financial data using candlestick chart with n-day candles.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Stock price DataFrame indexed by Date. Must contain columns:
        'Open', 'High', 'Low', 'Close', and optionally 'Volume'.
    
    n_days : int, default=1
        Number of trading days to aggregate into each candlestick.
        - n_days=1: Daily candles (default)
        - n_days=5: Weekly candles 
        - n_days>1: Multi-day aggregated candles
    
    title : str, default="Stock Price Candlestick Chart"
        Chart title to display.
    
    style : str, default='charles'
        mplfinance style theme. Options include:
        'charles', 'binance', 'blueskies', 'brasil', 'dark', 'default', etc.
    
    volume : bool, default=True
        Whether to display volume subplot below price chart.
    
    save_path : str or None, default=None
        If provided, saves the figure to this path (e.g., "figs/candlestick.png").
    
    figsize : tuple, default=(12, 8)
        Figure size as (width, height) in inches.
    
    How n-day aggregation works
    ---------------------------
    When n_days > 1, the function groups consecutive trading days:
    - Open: First day's opening price in the n-day period
    - High: Highest price across all n days  
    - Low: Lowest price across all n days
    - Close: Last day's closing price in the n-day period
    - Volume: Sum of volume across all n days
    
    Example: n_days=5 creates weekly candles where each candle represents
    5 consecutive trading days aggregated as described above.
    """
    
    # Input validation
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Handle n-day aggregation if n_days > 1
    if n_days > 1:
        print(f"Aggregating data into {n_days}-day candles...")
        
        # Group data into n-day chunks
        # We'll resample by grouping every n_days rows
        grouped_data = []
        
        # Iterate through data in chunks of n_days
        for i in range(0, len(data), n_days):
            chunk = data.iloc[i:i+n_days]
            if len(chunk) == 0:
                continue
                
            # Aggregate the chunk according to OHLC rules
            aggregated_row = {
                'Open': chunk['Open'].iloc[0],        # First day's open
                'High': chunk['High'].max(),          # Highest high
                'Low': chunk['Low'].min(),            # Lowest low  
                'Close': chunk['Close'].iloc[-1],     # Last day's close
            }
            
            # Add volume if present
            if 'Volume' in chunk.columns:
                aggregated_row['Volume'] = chunk['Volume'].sum()  # Sum of volumes
            
            # Use the last date in the chunk as the index
            aggregated_row['Date'] = chunk.index[-1]
            grouped_data.append(aggregated_row)
        
        # Create new DataFrame from aggregated data
        if grouped_data:
            data = pd.DataFrame(grouped_data)
            data.set_index('Date', inplace=True)
            print(f"Aggregated from {len(df)} daily records to {len(data)} {n_days}-day candles")
        else:
            raise ValueError("Not enough data to create any n-day candles")
    
    # Configure plot parameters
    plot_config = {
        'type': 'candle',           # Candlestick chart type
        'style': style,             # Visual style theme
        'title': title,             # Chart title
        'figsize': figsize,         # Figure dimensions
        'volume': volume,           # Show/hide volume subplot
    }
    
    # Additional styling options
    market_colors = mpf.make_marketcolors(
        up='green',         # Color for bullish (up) candles
        down='red',         # Color for bearish (down) candles  
        edge='inherit',     # Edge color inherits from up/down colors
        wick={'up':'green', 'down':'red'},  # Wick colors
        volume='in'         # Volume color matches candle color
    )
    
    # Create custom style with our color scheme
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors,
        gridstyle='-',      # Grid line style
        y_on_right=True     # Y-axis labels on right side
    )
    
    # Override style if using custom colors
    if style == 'custom':
        plot_config['style'] = custom_style
    
    # Handle save functionality
    if save_path:
        plot_config['savefig'] = {
            'fname': save_path,
            'dpi': 150,
            'bbox_inches': 'tight'
        }
    
    try:
        # Create the candlestick plot
        mpf.plot(data, **plot_config)
        print(f"Candlestick chart created successfully with {n_days}-day candles")
        
    except Exception as e:
        print(f"Error creating candlestick chart: {str(e)}")
        raise

def plot_boxplots_moving_window(
    df,
    price_column="Close",
    window=20,
    stride=1,
    showfliers=False,
    title="Rolling Window Boxplots",
    save_path=None
):
    """
    Display boxplots of a price column over rolling windows of 'window' trading days.

    Parameters
    ----------
    df : pandas.DataFrame
        Price DataFrame indexed by Date. Must include `price_column`.

    price_column : str, default='Close'
        Which column to summarize (e.g., 'Close', 'Open', 'High', 'Low').

    window : int, default=20
        Number of consecutive trading days per box (moving window length).

    stride : int, default=1
        Step size between windows (e.g., stride=5 plots every 5th window to reduce clutter).

    showfliers : bool, default=False
        Whether to show outliers on the boxplots.

    title : str
        Chart title.

    save_path : str or None
        If provided, saves the figure to this path (e.g., "figs/boxplot_win20.png").

    How it works
    ------------
    - For i from window to len(df), we take df[price_column][i-window:i] as one window.
    - Slide forward by `stride` rows each time.
    - Each window becomes one box in the boxplot, labeled by the window's end date.
    """

    if price_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_column}' column.")
    if window < 1:
        raise ValueError("window must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    prices = df[price_column].dropna()
    if len(prices) < window:
        raise ValueError("Not enough data to form one window.")

    # Collect rolling windows and labels
    data_windows = []
    labels = []
    idx = prices.index

    # Build each rolling window by slicing the Series
    for end in range(window, len(prices) + 1, stride):
        start = end - window
        w = prices.iloc[start:end].values
        data_windows.append(w)
        labels.append(idx[end - 1].strftime("%Y-%m-%d"))  # use end date as label

    # Plot boxplots
    plt.figure()
    plt.boxplot(
        data_windows,
        showfliers=showfliers,   # whether to draw outliers
        widths=0.6               # width of each box
    )
    plt.title(f"{title} (window={window}, stride={stride})")
    plt.ylabel(price_column)

    # Avoid clutter on the x-axis by showing ~10 evenly spaced labels
    if len(labels) > 10:
        step = max(1, len(labels) // 10)
        xticks = range(1, len(labels) + 1, step)
        xtick_labels = [labels[i - 1] for i in xticks]
        plt.xticks(xticks, xtick_labels, rotation=45, ha="right")
    else:
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------

# Define parameters for data loading
COMPANY = 'CBA.AX'           # Commonwealth Bank of Australia stock
TRAIN_START = '2021-01-01'   # Training data start date
TRAIN_END = '2025-08-01'     # Training data end date
PRICE_VALUE = 'Close'        # Use closing price for predictions
PREDICTION_DAYS = 60         # Use 60 days of history to predict next day

# Load and prepare the data
data, x_train, y_train, x_test, y_test, scalers = load_data(
    company=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    price_column=PRICE_VALUE,
    prediction_days=PREDICTION_DAYS
)

# Load original (unscaled) data for visualization
original_data, _, _, _, _, _ = load_data(
    company=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    scale=False,  # Don't scale for visualization
    prediction_days=PREDICTION_DAYS
)

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------

# Simple function to create different model architectures
def create_model(sequence_length, n_features, units=50, cell=LSTM, n_layers=3, 
                dropout=0.2, loss="mean_squared_error", optimizer="adam", bidirectional=False):
    """
    Create a deep learning model for stock prediction.
    
    This function creates different types of RNN models (LSTM, GRU, SimpleRNN) with
    configurable architecture. Based on the example code you provided.
    
    Parameters:
    -----------
    sequence_length: int
        Length of input sequences (e.g., 60 days)
    n_features: int  
        Number of features (e.g., 1 for just closing price)
    units: int
        Number of neurons in each layer (default: 50)
    cell: keras layer
        Type of RNN cell (LSTM, GRU, or SimpleRNN)
    n_layers: int
        Number of RNN layers (default: 3)
    dropout: float
        Dropout rate for regularization (default: 0.2)
    loss: str
        Loss function to use (default: "mean_squared_error")
    optimizer: str
        Optimizer to use (default: "adam")
    bidirectional: bool
        Whether to use bidirectional layers (default: False)
        
    Returns:
    --------
    model: Sequential
        Compiled Keras model ready for training
    """
    
    model = Sequential()
    
    # Add layers based on n_layers parameter
    for i in range(n_layers):
        if i == 0:
            # First layer needs input shape specification
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), 
                                      input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, 
                              input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            # Last RNN layer doesn't return sequences
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # Hidden layers return sequences
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        
        # Add dropout after each RNN layer to prevent overfitting
        model.add(Dropout(dropout))
    
    # Output layer - single neuron for price prediction
    model.add(Dense(1, activation="linear"))
    
    # Compile the model
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model

# Get the scaler for later use
scaler = scalers['Close']

# Create the model using our function (similar to original)
print("Creating LSTM model...")
model = create_model(
    sequence_length=x_train.shape[1],  # 60 days
    n_features=x_train.shape[2],       # 1 feature (Close price)
    units=50,                          # 50 neurons per layer
    cell=LSTM,                         # Use LSTM cells
    n_layers=3,                        # 3 layers (like original)
    dropout=0.2,                       # 20% dropout
    optimizer="adam"                   # Adam optimizer
)

print("Model architecture:")
model.summary()

# model = Sequential() # Basic neural network
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# Experiment 1: Try GRU instead of LSTM
print("\nExperiment 1: Creating GRU model...")
gru_model = create_model(
    sequence_length=x_train.shape[1],
    n_features=x_train.shape[2],
    units=50,
    cell=GRU,                          # Use GRU instead of LSTM
    n_layers=3,
    dropout=0.2,
    optimizer="adam"
)

# Experiment 2: Try Bidirectional LSTM
print("\nExperiment 2: Creating Bidirectional LSTM model...")
bi_lstm_model = create_model(
    sequence_length=x_train.shape[1],
    n_features=x_train.shape[2],
    units=50,
    cell=LSTM,
    n_layers=2,                        # Use fewer layers for bidirectional
    dropout=0.2,
    optimizer="adam",
    bidirectional=True                 # Make it bidirectional
)

# Experiment 3: Try different hyperparameters
print("\nExperiment 3: Creating deeper LSTM model...")
deep_model = create_model(
    sequence_length=x_train.shape[1],
    n_features=x_train.shape[2],
    units=100,                         # More neurons
    cell=LSTM,
    n_layers=4,                        # More layers
    dropout=0.3,                       # More dropout
    optimizer="rmsprop"                # Different optimizer
)

# This code will only run when the script is executed directly, not when imported
if __name__ == "__main__":
    # Train the original model (keeping the same training as before)
    print("\nTraining the main LSTM model...")
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Optional: Train other models for comparison
    print("\nTraining GRU model...")
    gru_model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)

    print("\nTraining Bidirectional LSTM model...")
    bi_lstm_model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)

    print("\nAll models trained successfully!")

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------
    # Load the test data
    TEST_START = '2023-08-02'
    TEST_END = '2024-07-02'

    # test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

    test_data = yf.download(COMPANY,TEST_START,TEST_END)


    # The above bug is the reason for the following line of code
    # test_data = test_data[1:]

    actual_prices = test_data[PRICE_VALUE].values

    total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    # We need to do the above because to predict the closing price of the fisrt
    # PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period

    model_inputs = model_inputs.reshape(-1, 1)
    # TO DO: Explain the above line

    model_inputs = scaler.transform(model_inputs)
    # We again normalize our closing price data to fit them into the range (0,1)
    # using the same scaler used above 
    # However, there may be a problem: scaler was computed on the basis of
    # the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
    # but there may be a lower/higher price during the test period 
    # [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
    # greater than one)
    # We'll call this ISSUE #2

    # TO DO: Generally, there is a better way to process the data so that we 
    # can use part of it for training and the rest for testing. You need to 
    # implement such a way

    #------------------------------------------------------------------------------
    # Make predictions on test data
    #------------------------------------------------------------------------------
    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # TO DO: Explain the above 5 lines

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Compare predictions from different models
    gru_predictions = gru_model.predict(x_test)
    gru_predictions = scaler.inverse_transform(gru_predictions)

    bi_predictions = bi_lstm_model.predict(x_test)
    bi_predictions = scaler.inverse_transform(bi_predictions)

    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 
    #------------------------------------------------------------------------------
    # Plot the test predictions
    ## To do:
    # 1) Candle stick charts
    # 2) Chart showing High & Lows of the day
    # 3) Show chart of next few days (predicted)
    #------------------------------------------------------------------------------

    # Debug: Print prediction values to check if they're valid
    print(f"Actual prices range: {actual_prices.min():.2f} to {actual_prices.max():.2f}")
    print(f"LSTM predictions range: {predicted_prices.min():.2f} to {predicted_prices.max():.2f}")
    print(f"GRU predictions range: {gru_predictions.min():.2f} to {gru_predictions.max():.2f}")
    print(f"Bi-LSTM predictions range: {bi_predictions.min():.2f} to {bi_predictions.max():.2f}")

    # Check if predictions are reasonable
    if predicted_prices.max() < 1:
    print("⚠️  LSTM predictions seem too small, checking raw values...")
    print(f"   Raw LSTM predictions: {predicted_prices[:5].flatten()}")
    
    if gru_predictions.max() < 1:
    print("⚠️  GRU predictions seem too small, checking raw values...")
    print(f"   Raw GRU predictions: {gru_predictions[:5].flatten()}")
    
    if bi_predictions.max() < 1:
    print("⚠️  Bi-LSTM predictions seem too small, checking raw values...")
    print(f"   Raw Bi-LSTM predictions: {bi_predictions[:5].flatten()}")

    plt.figure(figsize=(12, 8))
    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price", linewidth=2)

    # Always plot predictions, but with different styles if they seem problematic
    plt.plot(predicted_prices, color="green", label=f"LSTM Predicted {COMPANY} Price", linewidth=1.5, alpha=0.8)
    plt.plot(gru_predictions, color="blue", label=f"GRU Predicted {COMPANY} Price", linewidth=1.5, alpha=0.8)
    plt.plot(bi_predictions, color="red", label=f"Bi-LSTM Predicted {COMPANY} Price", linewidth=1.5, alpha=0.8)

    plt.title(f"{COMPANY} Share Price Prediction Comparison")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set y-axis limits to ensure all lines are visible
    all_values = np.concatenate([actual_prices.flatten(), predicted_prices.flatten(), gru_predictions.flatten(), bi_predictions.flatten()])
    plt.ylim(all_values.min() * 0.95, all_values.max() * 1.05)

    plt.show()

    #------------------------------------------------------------------------------
    # Predict next day
    #------------------------------------------------------------------------------

    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    # Make predictions with all models
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"LSTM Prediction: {prediction[0][0]:.2f}")

    gru_prediction = gru_model.predict(real_data)
    gru_prediction = scaler.inverse_transform(gru_prediction)
    print(f"GRU Prediction: {gru_prediction[0][0]:.2f}")

    bi_prediction = bi_lstm_model.predict(real_data)
    bi_prediction = scaler.inverse_transform(bi_prediction)
    print(f"Bidirectional LSTM Prediction: {bi_prediction[0][0]:.2f}")

    # Calculate average prediction
    avg_prediction = (prediction[0][0] + gru_prediction[0][0] + bi_prediction[0][0]) / 3
    print(f"Average Prediction: {avg_prediction:.2f}")

    # A few concluding remarks here:
    # 1. The predictor is quite bad, especially if you look at the next day 
    # prediction, it missed the actual price by about 10%-13%
    # Can you find the reason?
    # 2. The code base at
    # https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
    # gives a much better prediction. Even though on the surface, it didn't seem 
    # to be a big difference (both use Stacked LSTM)
    # Again, can you explain it?
    # A more advanced and quite different technique use CNN to analyse the images
    # of the stock price changes to detect some patterns with the trend of
    # the stock price:
    # https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
    # Can you combine these different techniques for a better prediction??

    #------------------------------------------------------------------------------
    # Task 1, 2, and 3: Advanced Prediction Functions
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Task 1, 2, and 3: Advanced Prediction Functions (Outside main block)
    #------------------------------------------------------------------------------

def create_multistep_sequences(data, price_column='Close', prediction_days=60, future_steps=5):
    """
    Task 1: Create sequences for multistep prediction (predict multiple days ahead).
    
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

def create_multivariate_sequences(data, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'], 
                                target_column='Close', prediction_days=60):
    """
    Task 2: Create sequences for multivariate prediction (multiple input features).
    
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

def create_multivariate_multistep_sequences(data, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                           target_column='Close', prediction_days=60, future_steps=5):
    """
    Task 3: Create sequences for multivariate, multistep prediction.
    
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
# Test the new functions
#------------------------------------------------------------------------------

print("\n" + "="*60)
print("TESTING NEW ADVANCED PREDICTION FUNCTIONS")
print("="*60)

# Test Task 1: Multistep Prediction
print("\nTask 1: Testing Multistep Prediction (5 days ahead)")
try:
    x_multistep, y_multistep = create_multistep_sequences(data, future_steps=5)
    print(f"SUCCESS: Multistep data created successfully!")
    print(f"   Input shape: {x_multistep.shape}")
    print(f"   Output shape: {y_multistep.shape}")
    
    # Create and test model
    multistep_model = create_multistep_model(x_multistep.shape[1], x_multistep.shape[2], 5)
    print(f"SUCCESS: Multistep model created successfully!")
    print(f"   Model expects input: {multistep_model.input_shape}")
    
except Exception as e:
    print(f"ERROR: Task 1 failed: {e}")

# Test Task 2: Multivariate Prediction
print("\nTask 2: Testing Multivariate Prediction (all features)")
try:
    # Check what columns we have
    available_cols = list(data.columns)
    feature_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in available_cols]
    print(f"   Available columns: {available_cols}")
    print(f"   Using features: {feature_cols}")
    
    if len(feature_cols) >= 2:  # Need at least 2 features
        x_multivar, y_multivar, scalers_multivar = create_multivariate_sequences(
            data, feature_columns=feature_cols
        )
        print(f"SUCCESS: Multivariate data created successfully!")
        print(f"   Input shape: {x_multivar.shape}")
        print(f"   Output shape: {y_multivar.shape}")
        print(f"   Features used: {len(feature_cols)}")
        
        # Create and test model
        multivar_model = create_multivariate_model(x_multivar.shape[1], x_multivar.shape[2])
        print(f"SUCCESS: Multivariate model created successfully!")
    else:
        print(f"WARNING: Not enough features available for multivariate prediction")
        
except Exception as e:
    print(f"ERROR: Task 2 failed: {e}")

# Test Task 3: Combined Multivariate + Multistep
print("\nTask 3: Testing Combined Multivariate + Multistep Prediction")
try:
    if len(feature_cols) >= 2:
        x_combined, y_combined, scalers_combined = create_multivariate_multistep_sequences(
            data, feature_columns=feature_cols, future_steps=3
        )
        print(f"SUCCESS: Combined data created successfully!")
        print(f"   Input shape: {x_combined.shape}")
        print(f"   Output shape: {y_combined.shape}")
        print(f"   Features: {len(feature_cols)}, Future steps: 3")
        
        # Create and test model
        combined_model = create_multivariate_multistep_model(
            x_combined.shape[1], x_combined.shape[2], 3
        )
        print(f"SUCCESS: Combined model created successfully!")
    else:
        print(f"WARNING: Not enough features for combined prediction")
        
except Exception as e:
    print(f"ERROR: Task 3 failed: {e}")

print("\n" + "="*60)
print("ADVANCED FUNCTIONS INTEGRATION COMPLETE!")
print("="*60)

#------------------------------------------------------------------------------
# Task 5: Ensemble Methods (v0.5)
# Combining ARIMA/SARIMA with Deep Learning Models
#------------------------------------------------------------------------------

def check_stationarity(timeseries):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    timeseries: array-like
        Time series data to test
        
    Returns:
    --------
    bool: True if stationary, False otherwise
    """
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    # If p-value < 0.05, series is stationary
    return result[1] < 0.05

def make_stationary(data, max_diff=3):
    """
    Make time series stationary by differencing.
    
    Parameters:
    -----------
    data: pandas Series
        Time series data
    max_diff: int
        Maximum number of differences to apply
        
    Returns:
    --------
    tuple: (stationary_data, n_diff)
    """
    data_diff = data.copy()
    n_diff = 0
    
    for i in range(max_diff):
        if check_stationarity(data_diff):
            print(f"Series is stationary after {n_diff} differences")
            break
        else:
            data_diff = data_diff.diff().dropna()
            n_diff += 1
            print(f"Applied difference {n_diff}")
    
    return data_diff, n_diff

def fit_arima_model(data, order=(1,1,1), seasonal_order=None):
    """
    Fit ARIMA or SARIMA model to time series data.
    
    Parameters:
    -----------
    data: pandas Series
        Time series data
    order: tuple
        (p,d,q) parameters for ARIMA
    seasonal_order: tuple
        (P,D,Q,s) parameters for SARIMA
        
    Returns:
    --------
    fitted_model: ARIMA or SARIMAX model
    """
    try:
        if seasonal_order is not None:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
            print(f"Fitting SARIMA{order}{seasonal_order} model...")
        else:
            model = ARIMA(data, order=order)
            print(f"Fitting ARIMA{order} model...")
            
        fitted_model = model.fit(disp=False)
        print(f"Model fitted successfully!")
        print(f"AIC: {fitted_model.aic:.2f}")
        return fitted_model
        
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None

def arima_predict(model, steps, start=None, end=None):
    """
    Make predictions using fitted ARIMA model.
    
    Parameters:
    -----------
    model: ARIMA/SARIMAX model
        Fitted model
    steps: int
        Number of steps to predict
    start: int, optional
        Start index for prediction
    end: int, optional
        End index for prediction
        
    Returns:
    --------
    predictions: pandas Series
        Predicted values
    """
    try:
        if start is not None and end is not None:
            predictions = model.predict(start=start, end=end)
        else:
            predictions = model.forecast(steps=steps)
        return predictions
    except Exception as e:
        print(f"Error making ARIMA predictions: {e}")
        return None

def create_random_forest_model(data, target_col='Close', n_estimators=100, max_depth=10):
    """
    Create and train Random Forest model for stock prediction.
    
    Parameters:
    -----------
    data: pandas DataFrame
        Stock data with features
    target_col: str
        Target column name
    n_estimators: int
        Number of trees in the forest
    max_depth: int
        Maximum depth of trees
        
    Returns:
    --------
    model: RandomForestRegressor
        Trained Random Forest model
    """
    try:
        # Create features (lagged values)
        features = []
        for i in range(1, 11):  # Use last 10 days as features
            data[f'lag_{i}'] = data[target_col].shift(i)
            features.append(f'lag_{i}')
        
        # Add technical indicators
        data['sma_5'] = data[target_col].rolling(window=5).mean()
        data['sma_20'] = data[target_col].rolling(window=20).mean()
        data['rsi'] = calculate_rsi(data[target_col])
        data['volatility'] = data[target_col].rolling(window=10).std()
        
        features.extend(['sma_5', 'sma_20', 'rsi', 'volatility'])
        
        # Remove NaN values
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            print("Not enough data for Random Forest")
            return None
            
        X = data_clean[features]
        y = data_clean[target_col]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        print(f"Random Forest trained successfully!")
        print(f"Training R²: {rf_model.score(X_train, y_train):.4f}")
        print(f"Test R²: {rf_model.score(X_test, y_test):.4f}")
        
        return rf_model, features
        
    except Exception as e:
        print(f"Error creating Random Forest model: {e}")
        return None, None

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    prices: pandas Series
        Price data
    window: int
        RSI calculation window
        
    Returns:
    --------
    rsi: pandas Series
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_ensemble_model(data, dl_model, arima_model=None, rf_model=None, 
                         ensemble_weights=None, method='weighted_average', 
                         x_test=None, scalers=None):
    """
    Create ensemble model combining multiple prediction methods.
    
    Parameters:
    -----------
    data: pandas DataFrame
        Stock data
    dl_model: keras model
        Trained deep learning model
    arima_model: ARIMA model, optional
        Trained ARIMA model
    rf_model: RandomForest model, optional
        Trained Random Forest model
    ensemble_weights: list, optional
        Weights for each model [dl_weight, arima_weight, rf_weight]
    method: str
        Ensemble method: 'weighted_average', 'voting', 'stacking'
        
    Returns:
    --------
    ensemble_predictions: array
        Ensemble predictions
    """
    try:
        predictions = []
        model_names = []
        
        # Deep Learning predictions
        if dl_model is not None:
            # Use test data for predictions
            dl_pred = dl_model.predict(x_test, verbose=0)
            dl_pred = scalers['Close'].inverse_transform(dl_pred)
            predictions.append(dl_pred.flatten())
            model_names.append('Deep Learning')
            print(f"Deep Learning predictions: {len(dl_pred)} samples")
        
        # ARIMA predictions
        if arima_model is not None:
            arima_pred = arima_predict(arima_model, steps=len(predictions[0]) if predictions else 100)
            if arima_pred is not None:
                predictions.append(arima_pred.values)
                model_names.append('ARIMA')
                print(f"ARIMA predictions: {len(arima_pred)} samples")
        
        # Random Forest predictions
        if rf_model is not None:
            # Prepare RF data
            rf_model_trained, features = rf_model
            if rf_model_trained is not None:
                # Create features for prediction using the same data as DL models
                # Use the test period data
                test_start_idx = len(data) - len(x_test) - 60  # Account for sequence length
                test_data = data.iloc[test_start_idx:].copy()
                
                # Create features
                for i in range(1, 11):
                    test_data[f'lag_{i}'] = test_data['Close'].shift(i)
                test_data['sma_5'] = test_data['Close'].rolling(window=5).mean()
                test_data['sma_20'] = test_data['Close'].rolling(window=20).mean()
                test_data['rsi'] = calculate_rsi(test_data['Close'])
                test_data['volatility'] = test_data['Close'].rolling(window=10).std()
                
                test_data_clean = test_data.dropna()
                if len(test_data_clean) >= len(predictions[0]) if predictions else 100:
                    X_pred = test_data_clean[features][:len(predictions[0]) if predictions else 100]
                    rf_pred = rf_model_trained.predict(X_pred)
                    predictions.append(rf_pred)
                    model_names.append('Random Forest')
                    print(f"Random Forest predictions: {len(rf_pred)} samples")
        
        if not predictions:
            print("No models available for ensemble")
            return None
        
        # Align prediction lengths
        min_length = min(len(pred) for pred in predictions)
        predictions = [pred[:min_length] for pred in predictions]
        
        # Create ensemble
        if method == 'weighted_average':
            if ensemble_weights is None:
                # Equal weights
                weights = [1.0/len(predictions)] * len(predictions)
            else:
                weights = ensemble_weights[:len(predictions)]
                weights = [w/sum(weights) for w in weights]  # Normalize
            
            ensemble_pred = np.zeros(min_length)
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            
            print(f"Ensemble weights: {dict(zip(model_names, weights))}")
            
        elif method == 'voting':
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)
            
        else:  # stacking
            # For simplicity, use weighted average
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred, predictions, model_names
        
    except Exception as e:
        print(f"Error creating ensemble model: {e}")
        return None, None, None

def evaluate_ensemble_performance(actual, predictions_dict, model_names):
    """
    Evaluate performance of ensemble and individual models.
    
    Parameters:
    -----------
    actual: array
        Actual values
    predictions_dict: dict
        Dictionary of predictions for each model
    model_names: list
        Names of models
        
    Returns:
    --------
    results: dict
        Performance metrics for each model
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    results = {}
    
    if actual is None:
        print("Warning: actual values are None, skipping evaluation")
        return results
    
    for name, pred in predictions_dict.items():
        if pred is not None and len(pred) == len(actual):
            mae = mean_absolute_error(actual, pred)
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, pred)
            
            # Directional accuracy
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(pred) > 0
            directional_acc = np.mean(actual_direction == pred_direction) * 100
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'Directional_Accuracy': directional_acc
            }
            
            print(f"{name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_acc:.2f}%")
    
    return results

def plot_ensemble_results(data, predictions_dict, model_names, save_path=None, actual_values=None):
    """
    Plot ensemble results comparing all models.
    
    Parameters:
    -----------
    data: pandas DataFrame
        Original stock data
    predictions_dict: dict
        Dictionary of predictions
    model_names: list
        Names of models
    save_path: str, optional
        Path to save the plot
    actual_values: array, optional
        Actual values to plot (if provided, use these instead of data)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time series comparison
    plt.subplot(2, 2, 1)
    
    # Get the length of predictions to align data
    pred_length = len(list(predictions_dict.values())[0])
    
    # Use provided actual_values if available, otherwise get from data
    if actual_values is not None:
        actual_data = actual_values
        # Create dates for the actual values (last pred_length days)
        actual_dates = data.index[-pred_length:]
        print(f"Using provided actual values: range {actual_data.min():.2f} to {actual_data.max():.2f}")
    else:
        # Get actual values from the test period (last pred_length days)
        actual_data = data['Close'].iloc[-pred_length:]
        actual_dates = data.index[-pred_length:]
        print(f"Using data actual values: range {actual_data.min():.2f} to {actual_data.max():.2f}")
    
    plt.plot(actual_dates, actual_data, 'b-', label='Actual', linewidth=2)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    for i, (name, pred) in enumerate(predictions_dict.items()):
        if len(pred) == pred_length:
            plt.plot(actual_dates, pred, 
                    color=colors[i % len(colors)], label=name, linewidth=1.5, alpha=0.8)
        else:
            print(f"Warning: {name} prediction length {len(pred)} doesn't match expected {pred_length}")
    
    plt.title('Ensemble Model Predictions vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Prediction errors
    plt.subplot(2, 2, 2)
    for i, (name, pred) in enumerate(predictions_dict.items()):
        if len(pred) == pred_length:
            errors = actual_data - pred
            plt.plot(errors, color=colors[i % len(colors)], label=f'{name} Error', alpha=0.7)
    
    plt.title('Prediction Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics comparison
    plt.subplot(2, 2, 3)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = ['MAE', 'RMSE', 'R²', 'Directional_Accuracy']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (name, pred) in enumerate(predictions_dict.items()):
        if len(pred) == pred_length:
            mae = mean_absolute_error(actual_data, pred)
            rmse = np.sqrt(mean_squared_error(actual_data, pred))
            r2 = r2_score(actual_data, pred)
            
            actual_direction = np.diff(actual_data) > 0
            pred_direction = np.diff(pred) > 0
            directional_acc = np.mean(actual_direction == pred_direction) * 100
            
            values = [mae, rmse, r2, directional_acc/100]  # Normalize directional accuracy
            plt.bar(x + i*width, values, width, label=name, color=colors[i % len(colors)], alpha=0.7)
    
    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(x + width*1.5, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of predictions vs actual
    plt.subplot(2, 2, 4)
    for i, (name, pred) in enumerate(predictions_dict.items()):
        if len(pred) == pred_length:
            plt.scatter(actual_data, pred, color=colors[i % len(colors)], 
                       label=name, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(actual_data.min(), min(pred.min() for pred in predictions_dict.values()))
    max_val = max(actual_data.max(), max(pred.max() for pred in predictions_dict.values()))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.title('Predictions vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ensemble results plot saved to: {save_path}")
    
    plt.show()

#------------------------------------------------------------------------------
# Task 5: Ensemble Experiments
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("TASK 5: ENSEMBLE METHODS - v0.5")
print("="*80)

# Load data for ensemble experiments
print("\n📊 Loading data for ensemble experiments...")
try:
    # Load data with longer history for better ARIMA fitting
    ensemble_data, x_train, y_train, x_test, y_test, scalers = load_data(
        company='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        prediction_days=60,
        test_size=0.2
    )
    
    print(f"✓ Loaded {len(ensemble_data)} days of data")
    print(f"✓ Training samples: {len(x_train)}")
    print(f"✓ Test samples: {len(x_test)}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    ensemble_data = None

if ensemble_data is not None:
    # Create output directory
    os.makedirs('ensemble_results', exist_ok=True)
    
    # 1. ARIMA Model
    print("\n🔍 Fitting ARIMA model...")
    try:
        # Check stationarity
        print("Checking stationarity...")
        is_stationary = check_stationarity(ensemble_data['Close'])
        
        if not is_stationary:
            print("Making data stationary...")
            stationary_data, n_diff = make_stationary(ensemble_data['Close'])
        else:
            stationary_data = ensemble_data['Close']
            n_diff = 0
        
        # Fit ARIMA model
        arima_model = fit_arima_model(stationary_data, order=(2, n_diff, 2))
        
    except Exception as e:
        print(f"❌ ARIMA model failed: {e}")
        arima_model = None
    
    # 2. Deep Learning Models
    print("\n🤖 Training Deep Learning models...")
    try:
        # LSTM Model
        lstm_model = create_model(
            sequence_length=x_train.shape[1],
            n_features=x_train.shape[2],
            units=50,
            cell=LSTM,
            n_layers=3,
            dropout=0.2
        )
        
        print("Training LSTM model...")
        lstm_history = lstm_model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # GRU Model
        gru_model = create_model(
            sequence_length=x_train.shape[1],
            n_features=x_train.shape[2],
            units=50,
            cell=GRU,
            n_layers=3,
            dropout=0.2
        )
        
        print("Training GRU model...")
        gru_history = gru_model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        print("✓ Deep Learning models trained successfully!")
        
    except Exception as e:
        print(f"❌ Deep Learning models failed: {e}")
        lstm_model = None
        gru_model = None
    
    # 3. Random Forest Model
    print("\n🌲 Training Random Forest model...")
    try:
        rf_model, rf_features = create_random_forest_model(
            ensemble_data, 
            n_estimators=100, 
            max_depth=10
        )
        print("✓ Random Forest model trained successfully!")
        
    except Exception as e:
        print(f"❌ Random Forest model failed: {e}")
        rf_model = None
    
    # 4. Create Ensemble Models
    print("\n🎯 Creating ensemble models...")
    
    # Define actual_values for all ensembles
    actual_values = None
    
    # Ensemble 1: LSTM + ARIMA
    if lstm_model is not None and arima_model is not None:
        print("\nCreating LSTM + ARIMA ensemble...")
        ensemble1_pred, ensemble1_components, ensemble1_names = create_ensemble_model(
            ensemble_data, 
            dl_model=lstm_model,
            arima_model=arima_model,
            ensemble_weights=[0.6, 0.4],  # Favor DL model
            method='weighted_average',
            x_test=x_test,
            scalers=scalers
        )
        
        if ensemble1_pred is not None:
            # Evaluate performance - use test data actual values
            actual_values = scalers['Close'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            ensemble1_dict = dict(zip(ensemble1_names, ensemble1_components))
            ensemble1_dict['LSTM+ARIMA Ensemble'] = ensemble1_pred
            
            print("\nLSTM + ARIMA Ensemble Performance:")
            ensemble1_results = evaluate_ensemble_performance(
                actual_values, ensemble1_dict, ensemble1_names
            )
            
            # Plot results
            plot_ensemble_results(
                ensemble_data, ensemble1_dict, ensemble1_names,
                save_path='ensemble_results/lstm_arima_ensemble.png',
                actual_values=actual_values
            )
    
    # Ensemble 2: LSTM + GRU + Random Forest
    if lstm_model is not None and gru_model is not None and rf_model is not None:
        print("\nCreating LSTM + GRU + Random Forest ensemble...")
        ensemble2_pred, ensemble2_components, ensemble2_names = create_ensemble_model(
            ensemble_data,
            dl_model=lstm_model,
            rf_model=(rf_model, rf_features),
            ensemble_weights=[0.4, 0.3, 0.3],  # Balanced weights
            method='weighted_average',
            x_test=x_test,
            scalers=scalers
        )
        
        if ensemble2_pred is not None:
            # Add GRU predictions manually
            gru_pred = gru_model.predict(x_test, verbose=0)
            gru_pred = scalers['Close'].inverse_transform(gru_pred)
            ensemble2_components.append(gru_pred.flatten())
            ensemble2_names.append('GRU')
            ensemble2_dict = dict(zip(ensemble2_names, ensemble2_components))
            ensemble2_dict['LSTM+GRU+RF Ensemble'] = ensemble2_pred
            
            # Get actual values for this ensemble - use test data actual values
            ensemble2_actual_values = scalers['Close'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            print("\nLSTM + GRU + Random Forest Ensemble Performance:")
            ensemble2_results = evaluate_ensemble_performance(
                ensemble2_actual_values, ensemble2_dict, ensemble2_names
            )
            
            # Plot results
            plot_ensemble_results(
                ensemble_data, ensemble2_dict, ensemble2_names,
                save_path='ensemble_results/lstm_gru_rf_ensemble.png',
                actual_values=ensemble2_actual_values
            )
    
    # Ensemble 3: All Models Combined
    if (lstm_model is not None and arima_model is not None and 
        gru_model is not None and rf_model is not None):
        print("\nCreating comprehensive ensemble (All Models)...")
        ensemble3_pred, ensemble3_components, ensemble3_names = create_ensemble_model(
            ensemble_data,
            dl_model=lstm_model,
            arima_model=arima_model,
            rf_model=(rf_model, rf_features),
            ensemble_weights=[0.3, 0.2, 0.2, 0.3],  # Balanced weights
            method='weighted_average',
            x_test=x_test,
            scalers=scalers
        )
        
        if ensemble3_pred is not None:
            # Add GRU predictions
            gru_pred = gru_model.predict(x_test, verbose=0)
            gru_pred = scalers['Close'].inverse_transform(gru_pred)
            ensemble3_components.append(gru_pred.flatten())
            ensemble3_names.append('GRU')
            ensemble3_dict = dict(zip(ensemble3_names, ensemble3_components))
            ensemble3_dict['All Models Ensemble'] = ensemble3_pred
            
            # Get actual values for this ensemble - use test data actual values
            ensemble3_actual_values = scalers['Close'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            print("\nComprehensive Ensemble Performance:")
            ensemble3_results = evaluate_ensemble_performance(
                ensemble3_actual_values, ensemble3_dict, ensemble3_names
            )
            
            # Plot results
            plot_ensemble_results(
                ensemble_data, ensemble3_dict, ensemble3_names,
                save_path='ensemble_results/comprehensive_ensemble.png',
                actual_values=ensemble3_actual_values
            )
    
    print("\n" + "="*80)
    print("✅ TASK 5: ENSEMBLE METHODS COMPLETE!")
    print("="*80)
    print("📊 Generated ensemble results and comparison plots")
    print("📁 Results saved in 'ensemble_results/' directory")
    print("🎯 Multiple ensemble combinations tested:")
    print("   • LSTM + ARIMA")
    print("   • LSTM + GRU + Random Forest") 
    print("   • All Models Combined")
    print("="*80)
