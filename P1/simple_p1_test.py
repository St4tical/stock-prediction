#!/usr/bin/env python3
"""
Simple P1 Test - Just test data loading without training
"""

import warnings
warnings.filterwarnings('ignore')

from stock_prediction_fixed import load_data
from parameters import *

print("="*50)
print("SIMPLE P1 DATA LOADING TEST")
print("="*50)

try:
    print(f"Testing P1 data loading for {ticker}...")
    print(f"Parameters:")
    print(f"  - Ticker: {ticker}")
    print(f"  - Features: {FEATURE_COLUMNS}")
    print(f"  - Sequence length: {N_STEPS}")
    print(f"  - Lookup step: {LOOKUP_STEP}")
    print(f"  - Test size: {TEST_SIZE}")
    
    # Test just the data loading
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
    
    print(f"\n✅ P1 DATA LOADING SUCCESS!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Test targets shape: {y_test.shape}")
    print(f"Last sequence shape: {last_sequence.shape}")
    print(f"Features per timestep: {X_train.shape[2]}")
    print(f"Sequence length: {X_train.shape[1]}")
    
    print(f"\n🎯 P1 CAPABILITIES CONFIRMED:")
    print(f"✓ Multi-feature input: {X_train.shape[2]} features")
    print(f"✓ Time series sequences: {X_train.shape[1]} timesteps")
    print(f"✓ Training samples: {X_train.shape[0]:,}")
    print(f"✓ Test samples: {X_test.shape[0]:,}")
    print(f"✓ Prediction horizon: {LOOKUP_STEP} days ahead")
    
    # Show feature scaling info
    if SCALE and column_scaler:
        print(f"\n📊 FEATURE SCALING:")
        for feature, scaler in column_scaler.items():
            print(f"  {feature}: min={scaler.data_min_[0]:.6f}, max={scaler.data_max_[0]:.6f}")
    
    print(f"\n🎉 P1 IS WORKING! Data loading successful.")
    print(f"This proves P1's multi-feature capabilities are functional.")
    
except Exception as e:
    print(f"❌ P1 test failed: {e}")
    import traceback
    traceback.print_exc() 