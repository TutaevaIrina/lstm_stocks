import os
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mi_lstm import MI_LSTM
from mi_dataset import generate_dataloaders

torch.manual_seed(0)

model_base_name = "MI_LSTM" # for storage later

# Hyperparameter
input_sizes = {
    "mainstream": 1,  # Pct_Gain
    "positive": 2,    # 40D_MA_to_Price_Ratio, 5D_MA_to_Price_Ratio
    "negative": 1,    # OBV_Ratio
    "index": 3        # RSI, ROC, Close
}
hidden_size = 128
num_layers = 2
output_size = 1
dropout_prob = 0.2
num_epochs = 10
learning_rate = 0.001
batch_size = 64
seq_length = 30

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the train and test loader
data_dir = "./Data"  # Ordner mit CSV-Dateien
models_dir = "./Models"  # Ordner mit Models-Dateien
train_loader, test_loader = generate_dataloaders(data_dir, seq_length=seq_length, batch_size=batch_size)

# Instantiate the model
model = MI_LSTM(input_sizes=input_sizes, hidden_size=hidden_size, num_layers=num_layers,
                output_size=output_size, dropout_prob=dropout_prob).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize loss lists
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

    for x_mainstream, x_positive, x_negative, x_index, targets in progress_bar:
        # Daten auf Gerät übertragen
        x_mainstream = x_mainstream.to(device)
        x_positive = x_positive.to(device)
        x_negative = x_negative.to(device)
        x_index = x_index.to(device)
        targets = targets.to(device)

        # Forward Pass
        optimizer.zero_grad()
        outputs = model(x_mainstream, x_positive, x_negative, x_index)

        # Loss calculaton
        loss = criterion(outputs, targets)
        loss.backward()

        # gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        progress_bar.set_postfix(train_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")

    # Test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_mainstream, x_positive, x_negative, x_index, targets in test_loader:
            x_mainstream = x_mainstream.to(device)
            x_positive = x_positive.to(device)
            x_negative = x_negative.to(device)
            x_index = x_index.to(device)
            targets = targets.to(device)

            outputs = model(x_mainstream, x_positive, x_negative, x_index)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.6f}")

    # Save model
    model_path = os.path.join("Models", f"{model_base_name}_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid()
plt.show()
