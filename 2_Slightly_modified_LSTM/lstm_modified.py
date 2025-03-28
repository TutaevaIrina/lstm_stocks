import torch
from torch import nn
import torch.nn.functional as F


#LSTM Model slightly modified
class LSTMModelModified(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout_prob=0.2):
        super(LSTMModelModified, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_prob
        )

        # Layer normalization after LSTM        
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)

        # Apply layer normalization on the outputs
        out = self.layer_norm(out[:, -1, :])  # Get output from the last time step

        # Pass through fully connected layers with dropout and activation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Output layer

        return out.squeeze()
