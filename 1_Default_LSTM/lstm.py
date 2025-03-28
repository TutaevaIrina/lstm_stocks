import torch
from torch import nn
import torch.nn.functional as F

# Basic LSTM Model, faster training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size = 50, num_layers = 2, output_size = 1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)

        # Initialize hidden and cell states with zeros       
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)

        # Get the last output of the LSTM for regression
        out = self.fc(out[:, -1, :])  # out: (batch_size, output_size)

        return out.squeeze()