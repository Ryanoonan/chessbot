import chess
import chess.svg
from neural_network import ChessEvalModel
from extra.minimax import minimax_search
import torch

class ChessAI:
    def __init__(self, model_path=None):
        self.board = chess.Board()
        self.model = ChessEvalModel()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode

    def get_best_move(self, depth=3):
        """Get the best move using minimax search with neural network evaluation."""
        move = minimax_search(self.board, depth, self.model)
        return move

    def make_move(self, move):
        """Make a move on the board."""
        self.board.push(move)

    def display_board(self):
        """Display the board."""
        return chess.svg.board(self.board)

    def get_game_state(self):
        """Return whether the game is over."""
        return self.board.is_game_over()
