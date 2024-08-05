import torch
import torch.nn as nn

# Define a simple bandwidth prediction model
class BandwidthPredictor(nn.Module):
    def __init__(self):
        super(BandwidthPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Load the model and use it for prediction
model = BandwidthPredictor()
model.load_state_dict(torch.load('data/models/bandwidth_model.pth'))  # Ensure the model weights are placed in the specified path
model.eval()

def predict_bandwidth(features):
    with torch.no_grad():
        features = torch.tensor(features, dtype=torch.float32)
        prediction = model(features)
        return prediction.item()
