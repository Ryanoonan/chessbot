import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.engine
import numpy as np
from chessnet import ChessNet, board_to_tensor
import random
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if STOCKFISH_PATH is None:
    STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"

class ChessDataset(Dataset):
    """Dataset of chess positions and their evaluations."""

    def __init__(self, num_positions=10000):  # Changed from 10000 to 100
        self.positions = []
        self.evaluations = []
        self.engine = chess.engine.SimpleEngine.popen_uci(
            STOCKFISH_PATH
        )

        # Generate random positions
        for _ in tqdm(range(num_positions), desc="Generating positions"):
            board = self._generate_random_position()
            evaluation = self._get_stockfish_evaluation(board)
            self.positions.append(board)
            self.evaluations.append(evaluation)

        self.engine.quit()

    def _generate_random_position(self) -> chess.Board:
        """Generate a random chess position."""
        board = chess.Board()
        num_moves = random.randint(0, 30)  # Random number of moves

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        return board

    def _get_stockfish_evaluation(self, board: chess.Board) -> float:
        """Get Stockfish evaluation for a position."""
        result = self.engine.analyse(board, chess.engine.Limit(time=0.3))
        score = result["score"].white().score(mate_score=10000)
        return score / 100.0  # Convert to centipawns

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = self.positions[idx]
        evaluation = self.evaluations[idx]

        # Convert board to tensor
        tensor = board_to_tensor(board)

        return tensor, torch.tensor([evaluation], dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """Train the chess evaluation model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
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
    dataset = ChessDataset(num_positions=10)  # Changed from 1000 to 100

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
