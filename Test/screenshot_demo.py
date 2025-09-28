#!/usr/bin/env python3
"""
Demo script to show data shape transformations for Task 5 Report screenshots
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create sample data for demonstration
print("=" * 60)
print("📊 TASK 5 REPORT - DATA SHAPE TRANSFORMATIONS DEMO")
print("=" * 60)

# Sample stock data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
sample_data = pd.DataFrame({
    'Date': dates,
    'Open': np.random.uniform(100, 110, 100),
    'High': np.random.uniform(110, 120, 100),
    'Low': np.random.uniform(90, 100, 100),
    'Close': np.random.uniform(95, 115, 100),
    'Volume': np.random.randint(1000000, 5000000, 100)
})

print(f"📈 Original Data Shape: {sample_data.shape}")
print(f"📈 Available Columns: {list(sample_data.columns)}")

# Demonstrate Task 1: Multistep Sequences
print("\n" + "=" * 40)
print("🎯 TASK 1: MULTISTEP PREDICTION")
print("=" * 40)

def demo_multistep_sequences(data, prediction_days=10, future_steps=3):
    price_data = data['Close'].values
    x_data, y_data = [], []
    
    for i in range(prediction_days, len(price_data) - future_steps + 1):
        x_data.append(price_data[i-prediction_days:i])
        y_data.append(price_data[i:i+future_steps])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data

x_multi, y_multi = demo_multistep_sequences(sample_data)
print(f"✅ Input Shape (X):  {x_multi.shape}  # (samples, time_steps, features)")
print(f"✅ Output Shape (Y): {y_multi.shape}  # (samples, future_steps)")
print(f"📊 Interpretation: {x_multi.shape[0]} samples, each using {x_multi.shape[1]} days to predict {y_multi.shape[1]} future days")

print("\n" + "=" * 60)
print("🎉 DEMO COMPLETE - READY FOR SCREENSHOTS!")
print("=" * 60) 