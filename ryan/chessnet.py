import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from typing import Tuple, Optional


class ChessNet(nn.Module):
    """Neural network for chess position evaluation, similar to Giraffe."""

    def __init__(self):
        super(ChessNet, self).__init__()
        # Fully connected layers
        self.relu = nn.ReLU()
        self.fc1w = nn.Linear(64*64*10, 256)
        self.fc1b = nn.Linear(64*64*10, 256)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 32)  # Output: single evaluation score
        self.fc4 = nn.Linear(32, 1)  # Output: single evaluation score



    def forward(self, x):
        # Flatten the input tensor
        # split x into two tensors: x1 and x2
        x1 = x[:, :, :, :, 0]  # White perspective
        x2 = x[:, :, :, :, 1]

        # Reshape to (batch_size, 8*8*10)
        x1 = x1.reshape(x1.size(0), -1)  # Reshape to (batch_size, 8*8*10)
        x2 = x2.reshape(x2.size(0), -1)  # Reshape to (batch_size, 8*8*10)

        # Pass through the first layer
        x1 = self.fc1w(x1)
        x2 = self.fc1b(x2)

        # Concatenate the outputs
        x = torch.cat((x1, x2), dim=1)  # Concatenate along the feature dimension
        x = self.relu(x)

        # Pass through the second and third layers
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)  # Output: single evaluation score

        return x





class GiraffeEvaluator:
    """Wrapper class for the neural network evaluation function."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = ChessNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def evaluate_position(self, board: chess.Board, eval: bool = False) -> float:
        """Evaluate a chess position using the neural network.

        Returns a centipawn score from White's perspective.
        Positive values favor White, negative values favor Black.
        """
        # Convert board to tensor (now already has channel dimension)

        tensor = board_to_tensor(board)

        # Add batch dimension (but not channel dimension)
        with torch.no_grad():
            score = self.model(tensor.unsqueeze(0)).item()  # Becomes [1, 1, 8, 8]

        return score  # Convert to centipawns


def minimax_search(
    board: chess.Board,
    depth: int,
    evaluator: GiraffeEvaluator,
    maximizing_player: bool = True,
) -> Tuple[float, Optional[chess.Move]]:
    """Minimax search using neural network evaluation."""

    if depth == 0 or board.is_game_over():
        return evaluator.evaluate_position(board), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax_search(board, depth - 1, evaluator, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax_search(board, depth - 1, evaluator, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

        return min_eval, best_move


def get_best_move(
    board: chess.Board, depth: int, evaluator: GiraffeEvaluator
) -> chess.Move:
    """Get the best move for the current position using minimax search."""
    _, best_move = minimax_search(board, depth, evaluator, maximizing_player=board.turn)
    return best_move


if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    evaluator = GiraffeEvaluator()

    # Get best move with depth 3
    best_move = get_best_move(board, depth=3, evaluator=evaluator)
    print(f"Best move: {best_move}")
