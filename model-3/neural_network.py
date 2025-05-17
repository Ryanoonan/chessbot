import torch
import torch.nn as nn
import torch.optim as optim
import chess

class ChessEvalModel(nn.Module):
    def __init__(self):
        super(ChessEvalModel, self).__init__()
        # A simple fully connected model, can be replaced with a CNN for better results
        self.fc1 = nn.Linear(64, 128)  # 64 input features (one for each square)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output (evaluation)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def eval_position(self, board):
        """Converts a chess board to a feature vector and passes it through the model."""
        feature_vector = self.board_to_feature_vector(board)
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        return self.forward(feature_tensor)

    def board_to_feature_vector(self, board):
        """Convert a chess board to a feature vector."""
        feature_vector = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            feature_vector.append(self.piece_to_feature(piece))
        return feature_vector

    def piece_to_feature(self, piece):
        """Convert a piece to a numeric feature."""
        if piece is None:
            return 0
        piece_map = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        return piece_map.get(piece.piece_type, 0) * (1 if piece.color == chess.WHITE else -1)

    def train(self, train_data, epochs=5, batch_size=32, lr=0.001):
        """Train the model on (board, eval) pairs."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()  # Mean squared error for evaluation

        for epoch in range(epochs):
            for board, eval_value in train_data:
                optimizer.zero_grad()
                output = self.eval_position(board)
                loss = criterion(output, torch.tensor([eval_value], dtype=torch.float32))
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
