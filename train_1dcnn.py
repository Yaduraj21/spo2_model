import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample, find_peaks
from preprocessing import prepare_signal # Import your cleaning logic

# 1. Define the same architecture as in main.py
class MobileVitalNet(nn.Module):
    def __init__(self):
        super(MobileVitalNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * 150, 64), nn.ReLU(),
            nn.Linear(64, 2) # [BPM, SpO2]
        )
    def forward(self, x):
        return self.regressor(self.features(x))

# 2. Load and Prepare Data
def load_training_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Get the PPG data
    raw_signal = df.iloc[:, 0].values 
    
    # 2. Clean it (CRITICAL: cleaning helps find_peaks see the signal)
    cleaned_signal = prepare_signal(raw_signal, fs=30)
    
    windows = []
    labels = []
    
    # 3. Slide through the data
    # Increase the step size slightly to move faster through 1.2 million samples
    for i in range(0, len(cleaned_signal) - 300, 200): 
        window = cleaned_signal[i:i+300]
        
        # --- IMPROVED PEAK DETECTION ---
        # distance=12: At 30fps, this allows up to 150 BPM (30/12 * 60)
        # height=0.1: Ensures we don't pick up tiny ripples of noise
        peaks, _ = find_peaks(window, distance=12, height=0.1)
        
        # If we find at least 5 peaks (a realistic amount for 10 seconds)
        if len(peaks) >= 5:
            # Calculate BPM: (number of beats / time in seconds) * 60
            duration_seconds = (peaks[-1] - peaks[0]) / 30
            bpm = ((len(peaks) - 1) / duration_seconds) * 60
            
            # Keep BPM within realistic human range
            if 40 < bpm < 180:
                windows.append(window)
                labels.append([bpm, 98.0]) 

    if len(windows) == 0:
        raise ValueError("❌ Still found 0 windows! Check if the CSV data is flat or empty.")

    print(f"✅ Successfully extracted {len(windows)} windows for training.")
    return torch.FloatTensor(np.array(windows)).unsqueeze(1), torch.FloatTensor(np.array(labels))

# 3. Training Loop
def train():
    X, y = load_training_data('processed_ppg_30fps.csv')
    model = MobileVitalNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Starting training on {len(X)} windows...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "vital_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    train()