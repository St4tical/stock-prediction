# Test script to create a corrupted CSV and test the auto-fixing feature

import pandas as pd
import os

def create_corrupted_csv():
    """Create a corrupted CSV file to test the auto-fixing feature."""
    
    print("🧪 Creating corrupted CSV file for testing...")
    
    # Create a sample corrupted CSV (like what we've been seeing)
    corrupted_data = """Date,Close,High,Low,Open,Volume
,CBA.AX,CBA.AX,CBA.AX,CBA.AX,CBA.AX
2021-01-04,69.53,69.67,68.21,68.26,1414844
2021-01-05,69.09,69.12,68.48,68.90,1809541
2021-01-06,68.82,69.19,68.30,68.65,2252786"""

    # Make sure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Write the corrupted data
    file_path = 'data/CBA.AX_2021-01-01_2025-08-01.csv'
    with open(file_path, 'w') as f:
        f.write(corrupted_data)
    
    print(f"✅ Created corrupted file: {file_path}")
    print("   This file contains 'CBA.AX' strings in numeric columns")
    print("   Your auto-fixing code should detect and fix this!")
    
    return file_path

def test_auto_fix():
    """Test the auto-fixing by importing and using the load_data function."""
    
    print("\n🔧 Testing auto-fix feature...")
    
    try:
        # Import the load_data function from your main code
        import sys
        sys.path.append('Code')
        from stock_prediction import load_data
        
        # Try to load data - this should trigger the auto-fix
        print("Calling load_data() - this should detect and fix corruption...")
        data, _, _, _, _, _ = load_data(
            company='CBA.AX',
            start_date='2021-01-01',
            end_date='2025-08-01',
            prediction_days=30,
            scale=False,
            test_size=0.1
        )
        
        print("✅ Auto-fix test completed!")
        print(f"   Final data shape: {data.shape}")
        print(f"   Data columns: {list(data.columns)}")
        
        # Verify data is clean
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors='raise')
                    print(f"   ✅ Column '{col}' is clean (all numeric)")
                except:
                    print(f"   ❌ Column '{col}' still has issues")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING AUTO-FIX CORRUPTION DETECTION")
    print("="*60)
    
    # Step 1: Create corrupted file
    corrupted_file = create_corrupted_csv()
    
    # Step 2: Test the auto-fix
    test_auto_fix()
    
    print("\n" + "="*60)
    print("🎉 AUTO-FIX TEST COMPLETE!")
    print("="*60)
    print("Your code should now automatically:")
    print("• Detect corrupted CSV files")
    print("• Delete corrupted files automatically") 
    print("• Download fresh data")
    print("• Clean any remaining bad data")
    print("• Continue working without manual intervention!") 