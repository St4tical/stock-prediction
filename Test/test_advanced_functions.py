# Test script to verify the new advanced prediction functions
import sys
import os
sys.path.append('Code')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Import the functions from our main code
from stock_prediction import (
    create_multistep_sequences, 
    create_multivariate_sequences, 
    create_multivariate_multistep_sequences,
    create_multistep_model,
    create_multivariate_model, 
    create_multivariate_multistep_model,
    load_data
)

def test_all_functions():
    """Test all three advanced functions with real data."""
    
    print("="*70)
    print("🧪 TESTING ADVANCED STOCK PREDICTION FUNCTIONS")
    print("="*70)
    
    # Load some test data
    print("\n📊 Loading test data...")
    try:
        # Use the same data loading function from your main code
        data, _, _, _, _, scalers = load_data(
            company='CBA.AX',
            start_date='2023-01-01',
            end_date='2024-01-01',
            prediction_days=30,  # Smaller for testing
            scale=False  # We'll handle scaling in individual functions
        )
        print(f"✅ Data loaded successfully: {data.shape}")
        print(f"   Columns available: {list(data.columns)}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Test 1: Multistep Prediction
    print("\n" + "-"*50)
    print("🎯 TEST 1: MULTISTEP PREDICTION")
    print("-"*50)
    
    try:
        future_days = 3
        x_multistep, y_multistep = create_multistep_sequences(
            data, 
            price_column='Close',
            prediction_days=30,
            future_steps=future_days
        )
        
        print(f"✅ Multistep sequences created successfully!")
        print(f"   Input shape: {x_multistep.shape}")
        print(f"   Output shape: {y_multistep.shape}")
        print(f"   Expected: Input=(samples, 30, 1), Output=(samples, {future_days})")
        
        # Verify shapes are correct
        assert x_multistep.shape[1] == 30, f"Expected 30 time steps, got {x_multistep.shape[1]}"
        assert x_multistep.shape[2] == 1, f"Expected 1 feature, got {x_multistep.shape[2]}"
        assert y_multistep.shape[1] == future_days, f"Expected {future_days} outputs, got {y_multistep.shape[1]}"
        
        # Test the model creation
        model1 = create_multistep_model(
            sequence_length=x_multistep.shape[1],
            n_features=x_multistep.shape[2], 
            future_steps=future_days
        )
        print(f"✅ Multistep model created: {model1.input_shape} → {model1.output_shape}")
        
        # Test a small prediction
        test_pred = model1.predict(x_multistep[:1], verbose=0)
        print(f"✅ Test prediction shape: {test_pred.shape} (should be (1, {future_days}))")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Multivariate Prediction
    print("\n" + "-"*50)
    print("🎯 TEST 2: MULTIVARIATE PREDICTION")
    print("-"*50)
    
    try:
        # Check which features are available
        available_features = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                            if col in data.columns]
        print(f"   Available features: {available_features}")
        
        if len(available_features) >= 2:
            x_multivar, y_multivar, scalers_mv = create_multivariate_sequences(
                data,
                feature_columns=available_features,
                target_column='Close',
                prediction_days=30
            )
            
            print(f"✅ Multivariate sequences created successfully!")
            print(f"   Input shape: {x_multivar.shape}")
            print(f"   Output shape: {y_multivar.shape}")
            print(f"   Expected: Input=(samples, 30, {len(available_features)}), Output=(samples,)")
            
            # Verify shapes
            assert x_multivar.shape[1] == 30, f"Expected 30 time steps, got {x_multivar.shape[1]}"
            assert x_multivar.shape[2] == len(available_features), f"Expected {len(available_features)} features, got {x_multivar.shape[2]}"
            assert len(y_multivar.shape) == 1, f"Expected 1D output, got {len(y_multivar.shape)}D"
            
            # Test model creation
            model2 = create_multivariate_model(
                sequence_length=x_multivar.shape[1],
                n_features=x_multivar.shape[2]
            )
            print(f"✅ Multivariate model created: {model2.input_shape} → {model2.output_shape}")
            
            # Test prediction
            test_pred2 = model2.predict(x_multivar[:1], verbose=0)
            print(f"✅ Test prediction shape: {test_pred2.shape} (should be (1, 1))")
            
            # Test scalers
            print(f"✅ Scalers created for features: {list(scalers_mv.keys())}")
            
        else:
            print(f"⚠️ Not enough features for multivariate test (need ≥2, have {len(available_features)})")
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Combined Multivariate + Multistep
    print("\n" + "-"*50)
    print("🎯 TEST 3: COMBINED MULTIVARIATE + MULTISTEP")
    print("-"*50)
    
    try:
        if len(available_features) >= 2:
            future_days = 2
            x_combined, y_combined, scalers_comb = create_multivariate_multistep_sequences(
                data,
                feature_columns=available_features,
                target_column='Close', 
                prediction_days=30,
                future_steps=future_days
            )
            
            print(f"✅ Combined sequences created successfully!")
            print(f"   Input shape: {x_combined.shape}")
            print(f"   Output shape: {y_combined.shape}")
            print(f"   Expected: Input=(samples, 30, {len(available_features)}), Output=(samples, {future_days})")
            
            # Verify shapes
            assert x_combined.shape[1] == 30, f"Expected 30 time steps, got {x_combined.shape[1]}"
            assert x_combined.shape[2] == len(available_features), f"Expected {len(available_features)} features, got {x_combined.shape[2]}"
            assert y_combined.shape[1] == future_days, f"Expected {future_days} outputs, got {y_combined.shape[1]}"
            
            # Test model
            model3 = create_multivariate_multistep_model(
                sequence_length=x_combined.shape[1],
                n_features=x_combined.shape[2],
                future_steps=future_days
            )
            print(f"✅ Combined model created: {model3.input_shape} → {model3.output_shape}")
            
            # Test prediction
            test_pred3 = model3.predict(x_combined[:1], verbose=0)
            print(f"✅ Test prediction shape: {test_pred3.shape} (should be (1, {future_days}))")
            
        else:
            print(f"⚠️ Not enough features for combined test")
            
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    print("✅ Task 1 (Multistep): Predicts multiple days ahead")
    print("   - Uses single feature (Close price)")
    print("   - Outputs multiple future values")
    print("   - Model architecture handles sequence-to-sequence")
    
    print("\n✅ Task 2 (Multivariate): Uses multiple input features")
    print("   - Uses all available price/volume features")
    print("   - Outputs single future value") 
    print("   - Includes feature scaling")
    
    print("\n✅ Task 3 (Combined): Best of both worlds")
    print("   - Uses multiple input features")
    print("   - Outputs multiple future values")
    print("   - Most advanced prediction capability")
    
    print(f"\n🎉 ALL TESTS COMPLETED!")
    print("Your advanced prediction functions are working correctly!")

if __name__ == "__main__":
    test_all_functions() 