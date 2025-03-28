import torch
from torch import nn
import torch.nn.functional as F

# Multi-Input LSTM Model with Attention
class MI_LSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size=128, num_layers=2, output_size=1, dropout_prob=0.2):     
        super(MI_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Separate LSTMs for each input factor
        self.lstm_mainstream = nn.LSTM(input_sizes["mainstream"], hidden_size, num_layers, batch_first=True)
        self.lstm_positive = nn.LSTM(input_sizes["positive"], hidden_size, num_layers, batch_first=True)
        self.lstm_negative = nn.LSTM(input_sizes["negative"], hidden_size, num_layers, batch_first=True)
        self.lstm_index = nn.LSTM(input_sizes["index"], hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_size * 4, 4)  # 4 factors
        self.softmax = nn.Softmax(dim=1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x_mainstream, x_positive, x_negative, x_index):
        batch_size = x_mainstream.size(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_mainstream.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_mainstream.device)

        # LSTM outputs for each factor
        out_mainstream, _ = self.lstm_mainstream(x_mainstream, (h0, c0))
        out_positive, _ = self.lstm_positive(x_positive, (h0, c0))
        out_negative, _ = self.lstm_negative(x_negative, (h0, c0))
        out_index, _ = self.lstm_index(x_index, (h0, c0))

        # Take the output from the last time step
        out_mainstream = out_mainstream[:, -1, :]  # (batch_size, hidden_size)
        out_positive = out_positive[:, -1, :]      # (batch_size, hidden_size)
        out_negative = out_negative[:, -1, :]      # (batch_size, hidden_size)
        out_index = out_index[:, -1, :]            # (batch_size, hidden_size)

        # Concatenate all outputs and compute attention weights
        concatenated = torch.cat((out_mainstream, out_positive, out_negative, out_index), dim=1)  # (batch_size, hidden_size * 4)
        attention_scores = self.attention_weights(concatenated)  # (batch_size, 4)
        attention_weights = self.softmax(attention_scores)  # Normalize to [0, 1]

        # Weighted sum of the outputs
        attended_out = (
            attention_weights[:, 0:1] * out_mainstream +
            attention_weights[:, 1:2] * out_positive +
            attention_weights[:, 2:3] * out_negative +
            attention_weights[:, 3:4] * out_index
        )  # (batch_size, hidden_size)

        # Apply layer normalization
        attended_out = self.layer_norm(attended_out)

        # Fully connected layers
        out = self.fc1(attended_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Output layer

        return out.squeeze()