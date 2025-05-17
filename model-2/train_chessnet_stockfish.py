import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.quantization
from torch.quantization import quantize_dynamic
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
    learning_rate: float = 0.001
):
    """Train the chess evaluation model."""
    # Check for device availability
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


def quantize_model(model, quantization_type='dynamic'):
    """
    Quantize the model weights to reduce memory usage
    
    Args:
        model: The trained PyTorch model
        quantization_type: Type of quantization ('dynamic', 'static', or 'qat')
        
    Returns:
        Quantized version of the model
    """
    if quantization_type == 'dynamic':
        # Dynamic quantization - weights are quantized to int8 but converted to float during computation
        # This is the simplest approach with minimal code changes
        quantized_model = quantize_dynamic(
            model=model,
            qconfig_spec={nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8  # Use 8-bit integers
        )
        return quantized_model
    
    elif quantization_type == 'static':
        # For 16-bit weights, we can use static quantization with float16
        model_fp16 = model.to(torch.float16)
        return model_fp16
    
    return model


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def inference_with_quantized_model(model_path, input_tensor, quantization_type='int8'):
    """Load and run inference with a quantized model"""
    model = ChessNet()
    
    if quantization_type == 'int8':
        # For 8-bit quantized model
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(model_path))
    elif quantization_type == 'fp16':
        # For 16-bit model
        model.load_state_dict(torch.load(model_path))
        model = model.to(torch.float16)
    
    # Set to evaluation mode
    model.eval()
    
    # Process the input tensor according to the quantization type
    if quantization_type == 'fp16':
        input_tensor = input_tensor.to(torch.float16)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output


def main():
    # Create dataset
    dataset = ChessDataset()
    #load dataset from lichess_dataset.pkl

    # Use this to filter the dataset
    # dataset = ChessDataset.load_from_file("lichess_dataset.pkl")
    # dataset.filter_noisy_positions(max_positions = 10000000)
    # dataset.save_to_file("filtered_lichess_dataset_100k_may15.pkl")


    dataset = ChessDataset.load_from_file("../data/lichess_dataset_2M_unique_no_mate.pkl")

    max_positions = 5000
    dataset.positions = dataset.positions[:max_positions]
    dataset.evaluations = dataset.evaluations[:max_positions]
    print(f"Loaded dataset with {len(dataset)} positions.")
    # Split into train and validation
    train_size = int(0.95 * len(dataset))
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

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
    
    # Save full precision model (already done in train_model, but here for clarity)
    torch.save(model.state_dict(), "chessnet_model_full.pth")
    
    # Create and save quantized models
    print("Creating quantized models...")
    
    # 8-bit quantized model
    quantized_model_int8 = quantize_model(model, quantization_type='dynamic')
    torch.save(quantized_model_int8.state_dict(), "chessnet_model_int8.pth")
    
    # 16-bit quantized model
    model_fp16 = model.to(torch.float16)
    torch.save(model_fp16.state_dict(), "chessnet_model_fp16.pth")
    
    # Print model sizes for comparison
    print(f"Original model size: {get_model_size_mb(model):.2f} MB")
    print(f"8-bit quantized model size: {get_model_size_mb(quantized_model_int8):.2f} MB")
    print(f"16-bit model size: {get_model_size_mb(model_fp16):.2f} MB")


if __name__ == "__main__":
    main()
