import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_length):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length

        # Group data by ticker and sort by date
        self.groups = data.groupby('Ticker')
        self.sequences = []

        for ticker, group in self.groups:
            # Sort by date
            group = group.sort_values('Date').reset_index(drop=True)

            # Get feature values and target values
            features = group[self.feature_cols].values
            targets = group[self.target_col].values

            # Create sequences
            for i in range(len(group) - self.seq_length):
                seq_features = features[i:i + self.seq_length]
                seq_target = targets[i + self.seq_length]  # Target is next step after the sequence
                self.sequences.append((seq_features, seq_target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_features, seq_target = self.sequences[idx]

        # Convert to torch tensors
        seq_features = torch.tensor(seq_features, dtype=torch.float32)
        seq_target = torch.tensor(seq_target, dtype=torch.float32)

        return seq_features, seq_target

def generate_dataloaders():

    print("generate_dataloaders: reading csv files and cleaning data")
    # Directory containing the CSV files
    data_dir = 'Data'

    # Initialize empty list for DataFrames
    df_list = []

    # Load data
    for filename in os.listdir(data_dir):
        if filename.endswith('_data.csv'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)

            # Add a column for the ticker
            ticker = filename.replace('_data.csv', '')
            df['Ticker'] = ticker

            df_list.append(df)

    # Concatenate all DataFrames
    data = pd.concat(df_list, ignore_index=True)

    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter for data after February 2020
    # This is done to avoid NaN, Inf and empty values caused by values that depend on previous values
    # such as 40 day moving average to price ratio (first proper values after 40 days from start of dataset)
    cutoff_date = '2020-02-01'
    data = data[data['Date'] >= cutoff_date].sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Check for inf values in numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    if np.isinf(numeric_data.values).any():
        print("generate_dataloaders: Inf values detected in the dataset. Rows with inf values:")
        print(data[np.isinf(numeric_data).any(axis=1)])

        # Replace inf values with 0
        # around 100 values out of 2 million rows contain nan which is just 0.005%
        # so replacing with 0 is absolutely fine
        data.replace([np.inf, -np.inf], 0, inplace=True)
        print("generate_dataloaders: Replaced inf values with 0.")

    # Identify tickers with NaN values and remove them
    # Also a small subset so removing them is also fine in this case
    tickers_with_nan = data[data.isna().any(axis=1)]['Ticker'].unique()
    if len(tickers_with_nan) > 0:
        print(f"generate_dataloaders: Removing tickers with NaN values: {tickers_with_nan}")
        data = data[~data['Ticker'].isin(tickers_with_nan)]

    # Check for NaN values after removing tickers
    if data.isna().any().any():
        print("generate_dataloaders: Unexpected NaN detected after removing tickers with NaNs. Rows with NaNs:")
        print(data[data.isna().any(axis=1)])

    # Define features and target
    # Keep Pct_Gain as feature?
    feature_cols = [col for col in data.columns if col not in ['Date', 'Ticker']]
    target_col = 'Pct_Gain'

    # Split into training and testing
    train_start_date = '2020-02-01'
    train_end_date = '2024-01-01'

    train_data = data[(data['Date'] >= train_start_date) & (data['Date'] < train_end_date)].reset_index(drop=True)
    test_data = data[data['Date'] >= train_end_date].reset_index(drop=True)

    # Create datasets
    seq_length = 30
    train_dataset = StockDataset(train_data, feature_cols, target_col, seq_length)
    test_dataset = StockDataset(test_data, feature_cols, target_col, seq_length)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, feature_cols