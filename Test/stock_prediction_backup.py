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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

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
        # Parse dates when loading to maintain proper datetime index
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    else:
        print(f"Downloading fresh data for {company} from {start_date} to {end_date}")
        # Download stock data using yfinance (more reliable than pandas_datareader)
        data = yf.download(company, start=start_date, end=end_date)
        
        # Save downloaded data for future use if enabled
        if save_locally:
            data.to_csv(file_path)
            print(f"Data saved to: {file_path}")

    # REQUIREMENT 1(b): NaN VALUE HANDLING
    # Handle missing values using specified method
    if fill_na_method == 'ffill':
        # Forward fill: use previous valid value to fill gaps
        data.fillna(method='ffill', inplace=True)

    elif fill_na_method == 'bfill':
        # Backward fill: use next valid value to fill gaps
        data.fillna(method='bfill', inplace=True)

    # REQUIREMENT 1(e): FEATURE SCALING WITH SCALER STORAGE
    # Dictionary to store fitted scalers for each feature column
    # This is crucial for inverse transformation later
    scalers = {}
    
    if scale:
        for col in feature_columns:
            # Create individual scaler for each feature column
            # MinMaxScaler normalizes data to range [0,1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Fit scaler to column data and transform it
            # reshape(-1, 1) converts 1D array to 2D column vector (required by sklearn)
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            
            # Store fitted scaler for future inverse transformations
            # This is essential for converting predictions back to original scale
            scalers[col] = scaler

    # LSTM SEQUENCE PREPARATION
    # LSTM networks need sequences of data to learn temporal patterns
    # We create sliding windows of 'prediction_days' length
    x_data, y_data = [], []
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
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

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
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

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