from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from preprocessing import prepare_signal

app = FastAPI()

# Architecture MUST match train_1dcnn.py exactly
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
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.regressor(self.features(x))

# Load model
model = MobileVitalNet()
model.load_state_dict(torch.load("vital_model.pth"))
model.eval()

class SignalInput(BaseModel):
    red_channel: list[float]

@app.post("/predict")
async def predict(data: SignalInput):
    # Process the signal exactly how we trained it
    signal = np.array(data.red_channel)
    if len(signal) != 300:
        return {"error": f"Expected 300 samples, got {len(signal)}"}
    
    clean_sig = prepare_signal(signal, fs=30)
    input_tensor = torch.FloatTensor(clean_sig).view(1, 1, -1)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        bpm, spo2 = prediction[0].tolist()
        
    return {
        "bpm": round(bpm, 1),
        "spo2": round(spo2, 1), # This will be 98.0 (our dummy value) for now
        "status": "Success"
    }