import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ChessEvalNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(ChessEvalNN, self).__init__()
        # Update the layer sizes to match the saved model
        self.layer1 = nn.Linear(12 * 8 * 8, 512)  # Match saved model size
        self.layer2 = nn.Linear(512, 256)         # Match saved model size
        self.output = nn.Linear(256, 1)           # Match saved model size

        self.dropout = nn.Dropout(dropout)  # Use dropout
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
    
        x = self.output(x)
        return x
