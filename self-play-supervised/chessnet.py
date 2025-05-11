
import torch
from torch import nn

class ChessNet(nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()
        # Increased network capacity with different activation functions
        self.fc1 = nn.Linear(12*64, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)  # Output: single evaluation score
        
        # Use Leaky ReLU instead of ReLU to allow negative values to propagate
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 12*64)
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation to allow both positive and negative outputs
        return x
