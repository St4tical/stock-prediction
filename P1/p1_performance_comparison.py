#!/usr/bin/env python3
"""
P1 Performance Comparison - Generate P1 prediction vs actual graph like v0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from stock_prediction_fixed import create_model, load_data
from parameters import *

print("="*70)
print("P1 PERFORMANCE COMPARISON - GENERATING P1 vs ACTUAL PREDICTION GRAPH")
print("="*70)

def create_p1_performance_graph():
    """Create P1 performance graph similar to v0.1 result"""
    
    print(f"Loading P1 data and model for {ticker}...")
    
    # Load data using P1's method
    X_train, X_test, y_train, y_test, column_scaler, last_sequence = load_data(
        ticker=ticker,
        n_steps=N_STEPS,
        scale=SCALE,
        split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS
    )
    
    print(f"✓ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create and load the pre-trained P1 model
    print("Loading pre-trained P1 model...")
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, 
                        cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, 
                        optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    
    # Load the pre-trained weights
    model_path = "results/2021-05-31_AMZN-sh-1-sc-1-sbd-0-huber_loss-adam-LSTM-seq-50-step-15-layers-2-units-256.h5"
    try:
        model.load_weights(model_path)
        print("✓ Pre-trained P1 model loaded successfully!")
        using_pretrained = True
    except:
        print("⚠️ Could not load pre-trained model, training new model...")
        # Quick training if pre-trained model fails
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
        using_pretrained = False
    
    # Make predictions
    print("Generating P1 predictions...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual prices
    if SCALE:
        y_test_actual = column_scaler["adjclose"].inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = column_scaler["adjclose"].inverse_transform(y_pred).flatten()
    else:
        y_test_actual = y_test
        y_pred_actual = y_pred.flatten()
    
    # Calculate performance metrics
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    rmse = np.sqrt(mse)
    
    # Calculate accuracy (within 5% tolerance)
    tolerance = 0.05
    accurate_predictions = np.abs((y_pred_actual - y_test_actual) / y_test_actual) <= tolerance
    accuracy = np.mean(accurate_predictions) * 100
    
    print(f"✓ P1 Performance Metrics:")
    print(f"  MSE: ${mse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  Accuracy (±5%): {accuracy:.1f}%")
    
    # Create the performance comparison graph (similar to v0.1)
    plt.figure(figsize=(14, 8))
    
    # Create time index for x-axis
    time_index = range(len(y_test_actual))
    
    # Plot actual vs predicted
    plt.plot(time_index, y_test_actual, label='Actual Price', color='blue', alpha=0.8, linewidth=2)
    plt.plot(time_index, y_pred_actual, label='P1 Predicted Price', color='red', alpha=0.8, linewidth=2)
    
    # Fill between for better visualization
    plt.fill_between(time_index, y_test_actual, y_pred_actual, alpha=0.2, color='gray')
    
    # Formatting
    plt.title(f'P1 Stock Prediction Performance - {ticker}\n'
              f'{"Pre-trained" if using_pretrained else "Newly Trained"} Model | '
              f'MAE: ${mae:.2f} | Accuracy: {accuracy:.1f}%', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance text box
    textstr = f'''P1 Model Performance:
    • Features: {len(FEATURE_COLUMNS)} (multi-feature)
    • Sequence Length: {N_STEPS} days
    • Prediction Horizon: {LOOKUP_STEP} days
    • Test Samples: {len(y_test_actual)}
    • RMSE: ${rmse:.2f}
    • MAE: ${mae:.2f}
    • Accuracy (±5%): {accuracy:.1f}%'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('p1_performance_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create a comparison with v0.1 style
    create_v01_style_comparison(y_test_actual, y_pred_actual, mae, accuracy, using_pretrained)
    
    return y_test_actual, y_pred_actual, mae, accuracy

def create_v01_style_comparison(y_test, y_pred, mae, accuracy, using_pretrained):
    """Create a graph in v0.1 style for direct comparison"""
    
    plt.figure(figsize=(12, 6))
    
    # Simple line plot like v0.1
    plt.plot(y_test, label='Actual Price', color='blue', linewidth=2)
    plt.plot(y_pred, label='P1 Predicted', color='red', linewidth=2)
    
    plt.title(f'P1 vs Actual Stock Prices (v0.1 Style Comparison)\n'
              f'MAE: ${mae:.2f} | Accuracy: {accuracy:.1f}%', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('p1_v01_style_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Generated both P1 performance graphs:")
    print("  - p1_performance_result.png (detailed)")
    print("  - p1_v01_style_result.png (v0.1 style)")

def load_existing_csv_results():
    """Try to load existing P1 CSV results if available"""
    
    csv_file = "csv-results/2021-05-31_AMZN-sh-1-sc-1-sbd-0-huber_loss-adam-LSTM-seq-50-step-15-layers-2-units-256.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded existing P1 results from CSV: {len(df)} records")
        
        # Extract actual and predicted prices
        actual_col = f'true_adjclose_{LOOKUP_STEP}'
        pred_col = f'adjclose_{LOOKUP_STEP}'
        
        if actual_col in df.columns and pred_col in df.columns:
            y_actual = df[actual_col].values
            y_predicted = df[pred_col].values
            
            # Remove any NaN values
            mask = ~(np.isnan(y_actual) | np.isnan(y_predicted))
            y_actual = y_actual[mask]
            y_predicted = y_predicted[mask]
            
            mae = np.mean(np.abs(y_actual - y_predicted))
            accuracy = np.mean(np.abs((y_predicted - y_actual) / y_actual) <= 0.05) * 100
            
            print(f"✓ CSV Results - MAE: ${mae:.2f}, Accuracy: {accuracy:.1f}%")
            
            # Create graph from CSV data
            plt.figure(figsize=(14, 8))
            plt.plot(y_actual, label='Actual Price', color='blue', linewidth=2)
            plt.plot(y_predicted, label='P1 Predicted (CSV)', color='red', linewidth=2)
            plt.title(f'P1 Performance from Saved Results\nMAE: ${mae:.2f} | Accuracy: {accuracy:.1f}%')
            plt.xlabel('Days')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('p1_csv_results_graph.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return True
    except Exception as e:
        print(f"Could not load CSV results: {e}")
        return False

if __name__ == "__main__":
    try:
        print("Attempting to load existing P1 CSV results first...")
        csv_loaded = load_existing_csv_results()
        
        if not csv_loaded:
            print("Generating new P1 performance results...")
            y_test, y_pred, mae, accuracy = create_p1_performance_graph()
        
        print("\n" + "="*70)
        print("P1 PERFORMANCE COMPARISON COMPLETE!")
        print("="*70)
        print("✅ Generated P1 performance graphs comparable to v0.1 result")
        print("✅ Now you have both v0.1 and P1 performance graphs for comparison")
        print("✅ This shows REAL P1 execution results, not simulation")
        
    except Exception as e:
        print(f"❌ P1 performance comparison failed: {e}")
        import traceback
        traceback.print_exc() 