# train_model.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from chess_model import ChessEvalNN  # Make sure to have your model in this file
from preprocess_data import ChessDataset  # Make sure to have the dataset class here
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# Instantiate the model
model = ChessEvalNN(dropout=0)  # With 30% dropout rate

# Loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
scheduler = StepLR(optimizer, step_size=5,gamma=0.1)

# Load the dataset (assuming you already have 'chess_dataset.csv' preprocessed)
dataset = ChessDataset("chess_dataset.csv")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

# Number of epochs (how many times to pass through the full dataset)
epochs = 100
best_val_loss = float('inf')  # Initialize the best validation loss

# Training loop with validation
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Training phase
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_batch)  # Forward pass
        loss = loss_fn(outputs, y_batch)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimization step
        running_loss += loss.item()  # Update the running loss
    
    # Calculate training loss for the current epoch
    train_loss = running_loss / len(train_dataloader)
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # No gradients needed during evaluation
        for X_val_batch, y_val_batch in val_dataloader:
            outputs = model(X_val_batch)  # Forward pass
            loss = loss_fn(outputs, y_val_batch)  # Calculate validation loss
            val_loss += loss.item()  # Update validation loss

    # Calculate validation loss for the current epoch
    val_loss = val_loss / len(val_dataloader)

    # Print the losses for the current epoch
    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}")

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_chess_model.pth')  # Save the model

    scheduler.step()
