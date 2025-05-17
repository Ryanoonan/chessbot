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
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # Output: single evaluation score


        # Initialize with reasonable piece values
        self._initialize_piece_values()

    def _initialize_piece_values(self):
        """Initialize the network with constant weights to approximate a sum of piece values."""
        # Standard piece values (in centipawns)


        # Set all weights to constant values
        with torch.no_grad():
            # Initialize fully connected layers with small constant weights
            nn.init.constant_(self.fc1.weight, 1/self.fc1.out_features)
            nn.init.constant_(self.fc2.weight, 1/self.fc2.out_features)
            nn.init.constant_(self.fc3.weight, 1/self.fc3.out_features)
            
            # Initialize fully connected biases
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)

            

    def forward(self, x):
        
        x = x.view(-1,self.fc1.in_features)  # Flatten the input tensor to shape (batch_size, 64)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        torch.clamp(x, min=-10, max=10)

        return x


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a chess board to a tensor representation.

    Returns a tensor of shape (8, 8) where:
    - Each cell contains the value of the piece in that cell:
      - White pieces: P=100, N=320, B=330, R=500, Q=900, K=20000
      - Black pieces: p=-100, n=-320, b=-330, r=-500, q=-900, k=-20000
      - Empty squares: 0
    """
    # Initialize empty tensor
    tensor = torch.zeros(8, 8)

    # Piece type to value mapping
    piece_values = {
        "P": 1.0,
        "N": 3.2,
        "B": 3.3,
        "R": 5.00,
        "Q": 9.00,
        "K": 200.00,
        "p": -1.00,
        "n": -3.20,
        "b": -3.30,
        "r": -5.00,
        "q": -9.00,
        "k": -200.00,
    }

    if board.is_checkmate():
        # Assign a large value for checkmate
        if board.turn == chess.WHITE:
            nn.init.constant(tensor, -20000)
        else:
            nn.init.constant(tensor, 20000)
        return tensor
    if board.is_stalemate() or board.is_insufficient_material():
        # Assign a value for stalemate
        nn.init.constant(tensor, 0)
        return tensor

    # Fill tensor with piece values
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.symbol()]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file] = value

    tensor = tensor.unsqueeze(0)  # Shape becomes (1, 8, 8)

    return tensor


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

        if eval:
            print("The tensor of the board is: ", tensor)

        # Add batch dimension (but not channel dimension)
        with torch.no_grad():
            score = self.model(tensor.unsqueeze(0)).item()  # Becomes [1, 1, 8, 8]

        return score  # Convert to centipawns


def minimax_search(
    board: chess.Board,
    depth: int,
    evaluator: GiraffeEvaluator,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing_player: bool = True,
) -> Tuple[float, Optional[chess.Move]]:
    """Minimax search with alpha-beta pruning using neural network evaluation."""

    if depth == 0 or board.is_game_over():
        return evaluator.evaluate_position(board), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax_search(board, depth - 1, evaluator, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax_search(board, depth - 1, evaluator, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

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
