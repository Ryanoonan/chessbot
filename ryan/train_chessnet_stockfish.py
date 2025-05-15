import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.engine
import numpy as np
from chessnet import ChessNet
import random
from tqdm import tqdm
import os
from dotenv import load_dotenv
import pickle

from generate_chess_dataset import ChessDataset

load_dotenv()

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if STOCKFISH_PATH is None:
    STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """Train the chess evaluation model."""
    # Check for M1/M2 Mac support first, then CUDA, then fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "chessnet_model.pth")
            print("Saved new best model!")


def main():
    # Create dataset
    dataset = ChessDataset()
    #load dataset from lichess_dataset.pkl

    # Use this to filter the dataset
    # dataset = ChessDataset.load_from_file("lichess_dataset.pkl")
    # dataset.filter_noisy_positions(max_positions = 10000000)
    # dataset.save_to_file("filtered_lichess_dataset_100k_may15.pkl")


    dataset = ChessDataset.load_from_file("filtered_lichess_dataset_1M_may15.pkl")

    max_positions = 20000
    dataset.positions = dataset.positions[:max_positions]
    dataset.evaluations = dataset.evaluations[:max_positions]
    print(f"Loaded dataset with {len(dataset)} positions.")
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = ChessNet()
    torch.save(model.state_dict(), "chessnet_untrained_model.pth")


    train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)


if __name__ == "__main__":
    main()
