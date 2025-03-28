import torch
from torch import nn
import torch.nn.functional as F    

class LSTMModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout_prob=0.2):
        super(LSTMModelWithAttention, self).__init__()
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

        # No need for a separate attention layer in this case

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state as the query vector
        h_T = hn[-1].unsqueeze(2)  # Shape: (batch_size, hidden_size, 1)

        # Compute attention scores as dot product between each output and h_T
        attn_scores = torch.bmm(out, h_T).squeeze(2)  # Shape: (batch_size, seq_length)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)  # Shape: (batch_size, seq_length, 1)

        # Compute context vector as weighted sum of LSTM outputs
        context = torch.sum(out * attn_weights, dim=1)  # Shape: (batch_size, hidden_size)

        # Pass through fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze()    