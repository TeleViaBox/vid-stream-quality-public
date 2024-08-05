import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# Create random data for training
X = torch.rand((100, 10))
y = torch.rand((100, 1))

# Create DataLoader
train_loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=True)

# Initialize model, loss function, and optimizer
model = BandwidthPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'data/models/bandwidth_model.pth')
