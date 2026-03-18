import pandas as pd
import numpy as np
from scipy.signal import resample
import os

def process_data():
    input_file = 'ppg_af_dataset.csv'
    output_file = 'processed_ppg_30fps.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    print("📖 Reading ppg_af_dataset.csv...")
    # Read the file and specifically look for the 'ppg' column
    df = pd.read_csv(input_file, usecols=['ppg'])
    
    # Remove any empty rows or non-numeric data
    df['ppg'] = pd.to_numeric(df['ppg'], errors='coerce')
    df = df.dropna()
    
    raw_signal = df['ppg'].values
    print(f"📊 Original signal found with {len(raw_signal)} samples.")

    print(f"🔄 Downsampling: 125Hz -> 30Hz...")
    target_len = int(len(raw_signal) * 30 / 125)
    
    # Perform resampling
    downsampled_signal = resample(raw_signal, target_len)
    
    # Final check: Replace any NaNs that resample might have created
    downsampled_signal = np.nan_to_num(downsampled_signal)
    
    # Save to CSV
    new_df = pd.DataFrame({'ppg': downsampled_signal})
    new_df.to_csv(output_file, index=False)
    
    print(f"✅ Success! Created {output_file}")
    print(f"📉 New sample count: {len(new_df)}")
    
    # Preview the first few rows to the console
    print("👀 Preview of processed data:")
    print(new_df.head())

if __name__ == "__main__":
    process_data()