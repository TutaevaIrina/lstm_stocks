import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class MultiInputStockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.seq_length = seq_length

        # Split features into input groups
        self.mainstream = data[['Pct_Gain']].values
        self.positive = data[['40D_MA_to_Price_Ratio', '5D_MA_to_Price_Ratio']].values
        self.negative = data[['OBV_Ratio']].values
        self.index = data[['RSI', 'ROC', 'Close']].values
        self.targets = data['Pct_Gain'].shift(-1).fillna(0).values  # Shift target for next step prediction

    def __len__(self):
        return len(self.targets) - self.seq_length

    def __getitem__(self, idx):
        # Extract sequences for each input group
        seq_mainstream = self.mainstream[idx:idx + self.seq_length]
        seq_positive = self.positive[idx:idx + self.seq_length]
        seq_negative = self.negative[idx:idx + self.seq_length]
        seq_index = self.index[idx:idx + self.seq_length]
        target = self.targets[idx + self.seq_length]

        return (
            torch.tensor(seq_mainstream, dtype=torch.float32),
            torch.tensor(seq_positive, dtype=torch.float32),
            torch.tensor(seq_negative, dtype=torch.float32),
            torch.tensor(seq_index, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


def generate_dataloaders(data_dir='../Data', seq_length=30, batch_size=64):   
    print("generate_dataloaders: Reading CSV files and preparing data")

    # Initialize empty list for DataFrames
    df_list = []

    # Load all data files
    for filename in os.listdir(data_dir):
        if filename.endswith('_data.csv'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)

            # Add ticker column
            ticker = filename.replace('_data.csv', '')
            df['Ticker'] = ticker
            df_list.append(df)

    # Combine all data
    data = pd.concat(df_list, ignore_index=True)

    # Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter out data before February 2020
    cutoff_date = '2020-02-01'
    data = data[data['Date'] >= cutoff_date].sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Replace infinite values with 0
    numeric_data = data.select_dtypes(include=[np.number])
    data.replace([np.inf, -np.inf], 0, inplace=True)

    # Remove rows with NaN values
    data.dropna(inplace=True)

    # Split into training and testing
    train_start_date = '2020-02-01'
    train_end_date = '2024-01-01'
    train_data = data[(data['Date'] >= train_start_date) & (data['Date'] < train_end_date)].reset_index(drop=True)
    test_data = data[data['Date'] >= train_end_date].reset_index(drop=True)

    # Create datasets
    train_dataset = MultiInputStockDataset(train_data, seq_length)
    test_dataset = MultiInputStockDataset(test_data, seq_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

