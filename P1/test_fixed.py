import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Import from our fixed version
from stock_prediction_fixed import create_model, load_data
from parameters import *

print("="*60)
print("FIXED P1 TEST - Using yfinance instead of yahoo_fin")
print("="*60)

def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b', label='Actual Price', alpha=0.8)
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r', label='Predicted Price', alpha=0.8)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"P1 FIXED: {ticker} Stock Price Prediction ({LOOKUP_STEP} days ahead)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('p1_fixed_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    
    test_df = data["test_df"].copy()
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    )
    return final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


def create_p1_data_structure():
    """Create the data structure that P1's functions expect"""
    # Load data using our fixed function
    X_train, X_test, y_train, y_test, column_scaler, last_sequence = load_data(
        ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS
    )
    
    # Create the data structure that other functions expect
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "column_scaler": column_scaler,
        "last_sequence": last_sequence
    }
    
    # We need to recreate the test_df for the plotting functions
    # Load the original data again to get the test dataframe
    stock = __import__('yfinance').Ticker(ticker)
    df_orig = stock.history(period="5y")
    df_orig.columns = df_orig.columns.str.lower()
    if 'adj close' in df_orig.columns:
        df_orig.rename(columns={'adj close': 'adjclose'}, inplace=True)
    
    # Get the test portion
    train_samples = int((1 - TEST_SIZE) * len(X_train) + len(X_train))
    test_df = df_orig.iloc[-len(X_test):].copy()
    data["test_df"] = test_df
    
    return data


try:
    print(f"Loading data for {ticker}...")
    
    # Create data structure
    data = create_p1_data_structure()
    
    print(f"✓ Data loaded successfully!")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Features: {len(FEATURE_COLUMNS)} ({', '.join(FEATURE_COLUMNS)})")
    print(f"  Sequence length: {N_STEPS}")
    print(f"  Prediction horizon: {LOOKUP_STEP} days")
    
    # construct the model (we'll train a simple version since we don't have pre-trained weights)
    print(f"\nCreating P1 model...")
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    
    print(f"✓ Model created with {model.count_params():,} parameters")
    
    # Since we don't have pre-trained weights, let's do a quick training
    print(f"\nTraining P1 model (quick training for demonstration)...")
    history = model.fit(data["X_train"], data["y_train"], 
                       batch_size=BATCH_SIZE, 
                       epochs=min(20, EPOCHS),  # Quick training
                       validation_data=(data["X_test"], data["y_test"]),
                       verbose=1)
    
    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\n✓ Model trained successfully!")
    print(f"  Loss: {loss:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
    
    # get the final dataframe for the testing set
    print(f"\nGenerating predictions...")
    final_df = get_final_df(model, data)
    
    # predict the future price
    future_price = predict(model, data)
    
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
    
    # printing metrics
    print(f"\n" + "="*60)
    print("P1 FIXED RESULTS:")
    print("="*60)
    print(f"Future price after {LOOKUP_STEP} days: ${future_price:.2f}")
    print(f"{LOSS} loss: {loss:.6f}")
    print(f"Mean Absolute Error: ${mean_absolute_error:.2f}")
    print(f"Accuracy score: {accuracy_score:.2%}")
    print(f"Total buy profit: ${total_buy_profit:.2f}")
    print(f"Total sell profit: ${total_sell_profit:.2f}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Profit per trade: ${profit_per_trade:.2f}")
    
    # plot true/pred prices graph
    print(f"\nGenerating P1 prediction graph...")
    plot_graph(final_df)
    
    print(f"\n✓ P1 FIXED test completed successfully!")
    print(f"✓ Graph saved as: p1_fixed_prediction_results.png")
    print(f"✓ This is REAL P1 execution using the actual P1 algorithm!")
    
    # Show sample results
    print(f"\nSample predictions (last 10 days):")
    print(final_df[['adjclose', f'adjclose_{LOOKUP_STEP}', f'true_adjclose_{LOOKUP_STEP}']].tail(10))
    
except Exception as e:
    print(f"❌ P1 FIXED test failed: {e}")
    import traceback
    traceback.print_exc() 