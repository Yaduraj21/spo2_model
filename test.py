import requests
import numpy as np

# Create a fake 10-second pulse (300 samples)
t = np.linspace(0, 10, 300)
# Simulate a 72 BPM pulse (1.2 Hz)
fake_pulse = np.sin(2 * np.pi * 1.2 * t).tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"red_channel": fake_pulse}
)

print("Response from Backend:", response.json())