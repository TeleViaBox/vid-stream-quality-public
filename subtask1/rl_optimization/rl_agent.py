import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])

        if done:
            target = reward
        else:
            with torch.no_grad():
                target = reward + 0.99 * torch.max(self.model(next_state))

        current_q = self.model(state)[action]
        loss = self.criterion(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
