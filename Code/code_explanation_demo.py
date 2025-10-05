#!/usr/bin/env python3
"""
Code Explanation Demo - Shows WHERE functions are and HOW they work
Creates screenshots of code locations and demonstrates data splitting
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

print("="*80)
print("CODE EXPLANATION DEMO - SHOWING FUNCTION LOCATIONS AND DATA SPLITTING")
print("="*80)

# Create output directory
os.makedirs('code_explanations', exist_ok=True)

# ============================================================================
# PART 1: CREATE CODE LOCATION SCREENSHOTS
# ============================================================================

print("\n📸 Creating code location screenshots...")

# Read the main file
with open('Code/stock_prediction.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Create function location map
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')

# Title
ax.text(0.5, 0.98, 'TASK 4: FUNCTION LOCATIONS IN stock_prediction.py',
        ha='center', va='top', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# Function locations
locations_text = """
FUNCTION LOCATIONS IN Code/stock_prediction.py:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATA LOADING & SPLITTING FUNCTION:
   Function: load_data()
   Location: Lines 44-292
   Purpose: Downloads stock data, handles NaN, scales features, splits train/test
   
   Key Parameters:
   • split_by_date=True/False  → Chronological vs Random split
   • test_size=0.2            → 20% for testing, 80% for training
   • company='CBA.AX'         → Stock ticker symbol
   • start_date, end_date     → Date range for data

🤖 DEEP LEARNING MODEL CREATION FUNCTION:
   Function: create_model()
   Location: Lines 577-646
   Purpose: Creates configurable DL models (LSTM, GRU, RNN, Bi-LSTM)
   
   Key Parameters:
   • cell=LSTM/GRU/SimpleRNN  → Type of RNN cell
   • n_layers=3               → Number of layers
   • units=50                 → Neurons per layer
   • bidirectional=False      → Whether to use bidirectional layers

🕯️ CANDLESTICK CHART FUNCTION:
   Function: plot_candlestick_chart()
   Location: Lines 300-448
   Purpose: Creates candlestick charts with n-day aggregation
   
   Key Parameters:
   • n_days=1                 → Days per candle (n ≥ 1)
   • df                      → Stock data with OHLC columns

📦 BOXPLOT CHART FUNCTION:
   Function: plot_boxplots_moving_window()
   Location: Lines 449-537
   Purpose: Creates boxplots for rolling windows of n days
   
   Key Parameters:
   • window=20                → n consecutive days per box
   • stride=1                 → Step size between windows

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE EXAMPLES:

1️⃣ CREATE DIFFERENT DL MODELS:
   model1 = create_model(sequence_length=60, n_features=1, 
                        units=50, cell=LSTM, n_layers=3)
   
   model2 = create_model(sequence_length=60, n_features=1,
                        units=100, cell=GRU, n_layers=2, bidirectional=True)

2️⃣ LOAD DATA WITH DIFFERENT SPLITS:
   # Chronological split (recommended for time series)
   data, x_train, y_train, x_test, y_test, scalers = load_data(
       company='AAPL', split_by_date=True, test_size=0.2)
   
   # Random split
   data, x_train, y_train, x_test, y_test, scalers = load_data(
       company='AAPL', split_by_date=False, test_size=0.3)

3️⃣ VISUALIZE DATA:
   plot_candlestick_chart(data, n_days=5, title="Weekly Candles")
   plot_boxplots_moving_window(data, window=20, stride=5)
"""

ax.text(0.02, 0.95, locations_text, ha='left', va='top', fontsize=10,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('code_explanations/FUNCTION_LOCATIONS.png', dpi=300, bbox_inches='tight')
print("✓ Saved: code_explanations/FUNCTION_LOCATIONS.png")
plt.close()

# ============================================================================
# PART 2: DEMONSTRATE DATA SPLITTING WITH DIFFERENT RATIOS
# ============================================================================

print("\n📊 Demonstrating data splitting with different ratios...")

# Download sample data
data = yf.download('AAPL', start='2022-01-01', end='2024-01-01', progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

print(f"✓ Downloaded {len(data)} days of AAPL data")

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

PREDICTION_DAYS = 30

def create_sequences(data, pred_days):
    x, y = [], []
    for i in range(pred_days, len(data)):
        x.append(data[i-pred_days:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_data, y_data = create_sequences(scaled_data, PREDICTION_DAYS)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

print(f"✓ Created {len(x_data)} sequences")

# Test different split ratios
split_ratios = [0.1, 0.2, 0.3, 0.4]
split_results = {}

for ratio in split_ratios:
    split_idx = int(len(x_data) * (1 - ratio))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    split_results[ratio] = {
        'train_samples': len(x_train),
        'test_samples': len(x_test),
        'train_percent': (len(x_train) / len(x_data)) * 100,
        'test_percent': (len(x_test) / len(x_data)) * 100,
        'split_idx': split_idx
    }
    
    print(f"  Test ratio {ratio}: Train={len(x_train)} ({len(x_train)/len(x_data)*100:.1f}%), Test={len(x_test)} ({len(x_test)/len(x_data)*100:.1f}%)")

# Create data splitting visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('DATA SPLITTING DEMONSTRATION - Different Test Ratios', 
             fontsize=16, fontweight='bold')

# Get dates for plotting
dates = data.index[PREDICTION_DAYS:]

for idx, (ratio, ax) in enumerate(zip(split_ratios, axes.flatten())):
    split_idx = split_results[ratio]['split_idx']
    
    # Plot training data
    ax.plot(dates[:split_idx], data['Close'].iloc[PREDICTION_DAYS:PREDICTION_DAYS+split_idx], 
           'b-', linewidth=2, label=f'Training Data ({split_results[ratio]["train_percent"]:.1f}%)', alpha=0.8)
    
    # Plot test data
    ax.plot(dates[split_idx:], data['Close'].iloc[PREDICTION_DAYS+split_idx:], 
           'r-', linewidth=2, label=f'Test Data ({split_results[ratio]["test_percent"]:.1f}%)', alpha=0.8)
    
    # Add split line
    ax.axvline(x=dates[split_idx], color='green', linestyle='--', linewidth=2, 
              label=f'Split Point ({dates[split_idx].strftime("%Y-%m-%d")})')
    
    ax.set_title(f'Test Ratio: {ratio} ({ratio*100:.0f}%)\nTrain: {split_results[ratio]["train_samples"]} samples, Test: {split_results[ratio]["test_samples"]} samples', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price ($)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('code_explanations/DATA_SPLITTING_DEMONSTRATION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: code_explanations/DATA_SPLITTING_DEMONSTRATION.png")
plt.close()

# ============================================================================
# PART 3: CREATE MODEL CREATION FUNCTION EXPLANATION
# ============================================================================

print("\n🤖 Creating model creation function explanation...")

fig, ax = plt.subplots(figsize=(16, 14))
ax.axis('off')

# Title
ax.text(0.5, 0.98, 'TASK 4: create_model() FUNCTION - DETAILED CODE EXPLANATION',
        ha='center', va='top', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# Code explanation
code_explanation = """
FUNCTION LOCATION: Code/stock_prediction.py, Lines 577-646

def create_model(sequence_length, n_features, units=50, cell=LSTM, n_layers=3, 
                dropout=0.2, loss="mean_squared_error", optimizer="adam", bidirectional=False):
    \"\"\"
    Create a deep learning model for stock prediction.
    
    This function creates different types of RNN models (LSTM, GRU, SimpleRNN) with
    configurable architecture. Based on the example code you provided.
    
    Parameters:
    -----------
    sequence_length: int        ← Length of input sequences (e.g., 60 days)
    n_features: int            ← Number of features (e.g., 1 for just closing price)
    units: int                 ← Number of neurons in each layer (default: 50)
    cell: keras layer          ← Type of RNN cell (LSTM, GRU, or SimpleRNN)
    n_layers: int              ← Number of RNN layers (default: 3)
    dropout: float             ← Dropout rate for regularization (default: 0.2)
    loss: str                  ← Loss function to use (default: "mean_squared_error")
    optimizer: str             ← Optimizer to use (default: "adam")
    bidirectional: bool        ← Whether to use bidirectional layers (default: False)
    \"\"\"
    
    model = Sequential()  ← Create empty model
    
    # Add layers based on n_layers parameter
    for i in range(n_layers):  ← Loop through each layer
        if i == 0:  ← FIRST LAYER
            # First layer needs input shape specification
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), 
                                      input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, 
                              input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:  ← LAST LAYER
            # Last RNN layer doesn't return sequences
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:  ← HIDDEN LAYERS
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO USE FOR DIFFERENT EXPERIMENTS:

1️⃣ DIFFERENT NETWORK TYPES:
   # LSTM Model
   lstm_model = create_model(60, 1, units=50, cell=LSTM, n_layers=3)
   
   # GRU Model  
   gru_model = create_model(60, 1, units=50, cell=GRU, n_layers=3)
   
   # SimpleRNN Model
   rnn_model = create_model(60, 1, units=50, cell=SimpleRNN, n_layers=3)
   
   # Bidirectional LSTM
   bi_model = create_model(60, 1, units=50, cell=LSTM, n_layers=3, bidirectional=True)

2️⃣ DIFFERENT LAYER COUNTS:
   # 2 Layers
   model_2layers = create_model(60, 1, units=50, cell=LSTM, n_layers=2)
   
   # 4 Layers
   model_4layers = create_model(60, 1, units=50, cell=LSTM, n_layers=4)
   
   # 5 Layers
   model_5layers = create_model(60, 1, units=50, cell=LSTM, n_layers=5)

3️⃣ DIFFERENT UNIT SIZES:
   # 25 units per layer
   model_25units = create_model(60, 1, units=25, cell=LSTM, n_layers=3)
   
   # 100 units per layer
   model_100units = create_model(60, 1, units=100, cell=LSTM, n_layers=3)
   
   # 256 units per layer
   model_256units = create_model(60, 1, units=256, cell=LSTM, n_layers=3)

4️⃣ DIFFERENT HYPERPARAMETERS:
   # Different dropout
   model_dropout = create_model(60, 1, units=50, cell=LSTM, n_layers=3, dropout=0.3)
   
   # Different optimizer
   model_rmsprop = create_model(60, 1, units=50, cell=LSTM, n_layers=3, optimizer="rmsprop")
   
   # Different loss function
   model_huber = create_model(60, 1, units=50, cell=LSTM, n_layers=3, loss="huber")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FEATURES:
✓ Flexible architecture - can create any combination of layers, units, cell types
✓ Supports LSTM, GRU, SimpleRNN, and Bidirectional variants
✓ Configurable dropout, optimizer, and loss function
✓ Returns compiled model ready for training
✓ Used in all Task 4 experiments to test different configurations
"""

ax.text(0.02, 0.95, code_explanation, ha='left', va='top', fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

plt.tight_layout()
plt.savefig('code_explanations/CREATE_MODEL_FUNCTION_EXPLANATION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: code_explanations/CREATE_MODEL_FUNCTION_EXPLANATION.png")
plt.close()

# ============================================================================
# PART 4: CREATE DATA SPLITTING FUNCTION EXPLANATION
# ============================================================================

print("\n📊 Creating data splitting function explanation...")

fig, ax = plt.subplots(figsize=(16, 14))
ax.axis('off')

# Title
ax.text(0.5, 0.98, 'DATA SPLITTING FUNCTION - DETAILED CODE EXPLANATION',
        ha='center', va='top', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

# Code explanation
split_explanation = """
FUNCTION LOCATION: Code/stock_prediction.py, Lines 252-285

# REQUIREMENT 1(c): FLEXIBLE TRAIN/TEST SPLITTING
if split_by_date:  ← CHRONOLOGICAL SPLIT (recommended for time series)
    # This respects temporal order - train on earlier data, test on later data
    
    # Calculate split index based on test_size
    if isinstance(test_size, float):  ← If test_size is percentage (e.g., 0.2)
        # If test_size is float (e.g., 0.2), treat as percentage
        split_idx = int(len(x_data) * (1 - test_size))  ← 80% for training
    else:  ← If test_size is absolute number (e.g., 100)
        # If test_size is int, treat as absolute number of test samples
        split_idx = len(x_data) - test_size
        
    # Split data chronologically
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]  ← First 80%, last 20%
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
else:  ← RANDOM SPLIT
    # Create array of indices and shuffle them randomly
    indices = np.arange(len(x_data))  ← [0, 1, 2, 3, ..., n]
    np.random.shuffle(indices)        ← [45, 12, 78, 3, ..., 23] (random order)
    
    # Calculate split point
    if isinstance(test_size, float):
        split_idx = int(len(x_data) * (1 - test_size))
    else:
        split_idx = len(x_data) - test_size
        
    # Split indices into train and test sets
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    # Use shuffled indices to create train/test sets
    x_train, x_test = x_data[train_idx], x_data[test_idx]  ← Random samples
    y_train, y_test = y_data[train_idx], y_data[test_idx]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW THE SPLITTING WORKS:

1️⃣ CHRONOLOGICAL SPLIT (split_by_date=True):
   Data: [Day1][Day2][Day3][Day4][Day5][Day6][Day7][Day8][Day9][Day10]
   
   test_size=0.2 (20% for testing):
   Train: [Day1][Day2][Day3][Day4][Day5][Day6][Day7][Day8]  ← First 80%
   Test:  [Day9][Day10]                                     ← Last 20%
   
   ✓ Preserves temporal order
   ✓ Train on past, test on future (realistic)
   ✓ Recommended for time series

2️⃣ RANDOM SPLIT (split_by_date=False):
   Data: [Day1][Day2][Day3][Day4][Day5][Day6][Day7][Day8][Day9][Day10]
   
   Shuffled indices: [Day3][Day7][Day1][Day9][Day5][Day2][Day8][Day4][Day6][Day10]
   
   test_size=0.2 (20% for testing):
   Train: [Day3][Day7][Day1][Day9][Day5][Day2][Day8]  ← Random 80%
   Test:  [Day4][Day6][Day10]                         ← Random 20%
   
   ✓ Mixes past and future data
   ✓ May cause data leakage
   ⚠ Not recommended for time series

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE EXAMPLES:

# Chronological split (recommended)
data, x_train, y_train, x_test, y_test, scalers = load_data(
    company='AAPL',
    split_by_date=True,     ← Use chronological split
    test_size=0.2          ← 20% for testing
)

# Random split
data, x_train, y_train, x_test, y_test, scalers = load_data(
    company='AAPL', 
    split_by_date=False,    ← Use random split
    test_size=0.3          ← 30% for testing
)

# Different test sizes
data, x_train, y_train, x_test, y_test, scalers = load_data(
    company='AAPL',
    test_size=0.1          ← 10% for testing (90% training)
)

data, x_train, y_train, x_test, y_test, scalers = load_data(
    company='AAPL',
    test_size=0.4          ← 40% for testing (60% training)
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY CHRONOLOGICAL SPLIT IS BETTER FOR STOCK PREDICTION:

✓ Realistic: Train on historical data, predict future
✓ No data leakage: Future data doesn't influence past predictions
✓ Time series nature: Stock prices have temporal dependencies
✓ Practical: Matches real-world trading scenarios

⚠ Random split problems:
✗ Data leakage: Future patterns leak into training
✗ Unrealistic: Can't use future data to predict past
✗ Overoptimistic: May give artificially good results
"""

ax.text(0.02, 0.95, split_explanation, ha='left', va='top', fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('code_explanations/DATA_SPLITTING_FUNCTION_EXPLANATION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: code_explanations/DATA_SPLITTING_FUNCTION_EXPLANATION.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ CODE EXPLANATION DEMO COMPLETE!")
print("="*80)

print("\n📸 Generated Code Explanation Files:")
print("  1. FUNCTION_LOCATIONS.png                    - Shows WHERE all functions are")
print("  2. DATA_SPLITTING_DEMONSTRATION.png          - Shows HOW data splits work")
print("  3. CREATE_MODEL_FUNCTION_EXPLANATION.png     - Detailed create_model() code")
print("  4. DATA_SPLITTING_FUNCTION_EXPLANATION.png   - Detailed splitting code")

print("\n🎯 KEY FINDINGS:")
print("  • create_model() function: Lines 577-646 in stock_prediction.py")
print("  • load_data() function: Lines 44-292 in stock_prediction.py")
print("  • Data splitting: Lines 252-285 in load_data() function")
print("  • Supports both chronological and random splits")
print("  • Flexible test_size parameter (0.1 to 0.4 tested)")

print("\n💡 FOR YOUR REPORT:")
print("  • Use FUNCTION_LOCATIONS.png to show WHERE functions are")
print("  • Use DATA_SPLITTING_DEMONSTRATION.png to show HOW splits work")
print("  • Use CREATE_MODEL_FUNCTION_EXPLANATION.png for Task 4 code details")
print("  • All functions are in Code/stock_prediction.py with clear line numbers")
print("="*80)
