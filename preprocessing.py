import numpy as np
from scipy.signal import butter, filtfilt

def prepare_signal(raw_data, fs=30):
    """
    Cleans the raw PPG signal (Red or Blue channel).
    fs: Sampling frequency (30 FPS for most phones)
    """
    # 1. Bandpass Filter: Remove high-freq noise and DC drift
    # Keeping human pulse range (0.5Hz = 30BPM to 4Hz = 240BPM)
    nyquist = 0.5 * fs
    low, high = 0.5 / nyquist, 4.0 / nyquist
    b, a = butter(3, [low, high], btype='band')
    
    # 2. Filter the signal
    cleaned = filtfilt(b, a, raw_data)
    
    # 3. Z-Score Normalization
    # Makes the signal independent of skin tone or flash brightness
    mean, std = np.mean(cleaned), np.std(cleaned)
    normalized = (cleaned - mean) / (std + 1e-8)
    
    return normalized