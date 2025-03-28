import os
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from dataset import generate_dataloaders
from lstm_modified import LSTMModelModified

torch.manual_seed(0)

model_base_name = "LSTMModelModified" # for storage later


# Get the train and test loader, and the feature columns
train_loader, test_loader, feature_cols = generate_dataloaders()

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and set hyperparameters
input_size = len(feature_cols)
num_epochs = 10
learning_rate = 0.001
model = LSTMModelModified(input_size = input_size, hidden_size=128, num_layers=2, output_size=1, dropout_prob=0.2).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize loss lists
train_losses = []
test_losses = []

# Train loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # Add tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Check for NaN or Inf in inputs and targets, just to be safe
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("NaN or Inf detected in inputs")
            continue
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("NaN or Inf detected in targets")
            continue

        optimizer.zero_grad()
        outputs = model(inputs)

        # Check for NaN in outputs, this is needed in case of exploding gradient issues
        if torch.isnan(outputs).any():
            print("NaN detected in outputs")
            continue

        loss = criterion(outputs, targets)

        # Check for NaN in loss (same reason as above)
        if torch.isnan(loss).any():
            print("NaN detected in loss. Stopping training.")
            break

        loss.backward()
        # gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        # Update tqdm description with the current loss
        progress_bar.set_postfix(train_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader.dataset)    
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')

    # Intermediate test loss calculation using a subset of test data
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.6f}')

    # Save model checkpoint
    model_path = os.path.join("Models", f"{model_base_name}_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# Plot Training- und Test-Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid()
plt.show()

    