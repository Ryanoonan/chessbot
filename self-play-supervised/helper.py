import chess
import pygame
import os

import torch

class ChessGame:
    def __init__(self, window_size=480):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(
            (window_size + 200, window_size)
        )  # Added 200 pixels for evaluation display
        pygame.display.set_caption("Chess: Human vs Giraffe")
        self.square_size = window_size // 8
        self.load_sprites()
        self.font = pygame.font.SysFont("Arial", 24)
        self.selected_square = None  # Track the currently selected square

    def load_sprites(self):
        """Load chess piece sprites."""
        self.piece_sprites = {}
        for piece in ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]:
            filename = f"{piece.lower()}{'d' if piece.islower() else 'l'}t60.png"
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets",
                filename,
            )
            image = pygame.image.load(path)
            self.piece_sprites[piece] = pygame.transform.scale(
                image, (self.square_size, self.square_size)
            )

    def draw_evaluation(self, evaluation):
        """Draw the evaluation bar on the right side of the board."""
        # Draw background
        pygame.draw.rect(
            self.screen, (200, 200, 200), (self.window_size, 0, 200, self.window_size)
        )

        # Draw evaluation text
        eval_text = f"Evaluation: {evaluation:.2f}"
        text_surface = self.font.render(eval_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (self.window_size + 10, 10))

        # Draw evaluation bar
        max_eval = 10.0  # Maximum evaluation to display
        bar_height = self.window_size - 40
        eval_height = (evaluation / max_eval) * (bar_height / 2)
        if evaluation > 0:
            # White advantage
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (
                    self.window_size + 10,
                    self.window_size / 2 - eval_height,
                    30,
                    eval_height,
                ),
            )
        else:
            # Black advantage
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (self.window_size + 10, self.window_size / 2, 30, -eval_height),
            )

    def draw_board(self, board, evaluation=None, highlight_moves=None):
        """Draw the chess board and pieces."""
        # Initialize highlight_moves if not provided
        if highlight_moves is None:
            highlight_moves = []
            
        # Draw squares
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size
                square = chess.square(col, 7 - row)
                
                # Default square colors
                if (row + col) % 2 == 0:
                    color = (255, 255, 255)  # Light squares
                else:
                    color = (128, 128, 128)  # Dark squares
                
                # Highlight selected square
                if square == self.selected_square:
                    color = (255, 255, 0)  # Yellow for selected
                
                # Highlight possible moves
                if square in highlight_moves:
                    color = (173, 216, 230)  # Light blue for possible moves
                
                pygame.draw.rect(
                    self.screen, color, (x, y, self.square_size, self.square_size)
                )

                # Draw piece if present
                piece = board.piece_at(square)
                if piece:
                    sprite = self.piece_sprites.get(piece.symbol())
                    if sprite:
                        self.screen.blit(sprite, (x, y))

        # Draw evaluation if provided
        if evaluation is not None:
            self.draw_evaluation(evaluation)

        pygame.display.flip()

    def display_board(self):
        """Display the current board position and evaluation."""
        print("\nCurrent position:")
        print(self.board)

        # Get evaluation from White's perspective
        evaluation = self.model.evaluate_position(self.board)
        if not self.board.turn:  # If it's Black's turn, negate the evaluation
            evaluation = -evaluation

        # Convert evaluation to pawn units and format the display
        eval_in_pawns = evaluation / 100.0  # Convert centipawns to pawns
        print(f"\nEvaluation: {eval_in_pawns:.2f} pawns")
        if eval_in_pawns > 0:
            print("White is better")
        elif eval_in_pawns < 0:
            print("Black is better")
        else:
            print("Position is equal")

        # Show whose turn it is
        print(f"\n{'White' if self.board.turn else 'Black'} to move")

        # Show valid moves
        valid_moves = [move.uci() for move in self.board.legal_moves]
        print("\nValid moves:", ", ".join(valid_moves))


# Define board_to_tensor function
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a chess board to a tensor representation for the neural network."""
    # Initialize tensor with 12 channels (6 piece types x 2 colors)
    tensor = torch.zeros(12, 8, 8)
    
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    # Fill tensor with piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get rank and file (0-7)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            # Set the corresponding channel to 1
            piece_channel = piece_idx[piece.symbol()]
            tensor[piece_channel, rank, file] = 1.0
            
    return tensor


def minimax_search(
    board: chess.Board,
    depth: int,
    model,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing_player: bool = True,
) -> tuple[float, chess.Move | None]:
    """Minimax search with alpha-beta pruning using neural network evaluation."""

    if depth == 0 or board.is_game_over():
        return model(board_to_tensor(board)), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax_search(board, depth - 1, model, alpha, beta, False)
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
            eval, _ = minimax_search(board, depth - 1, model, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_move


def get_best_move(
    board: chess.Board, depth: int, evaluator
) -> chess.Move:
    """Get the best move for the current position using minimax search."""
    _, best_move = minimax_search(board, depth, evaluator, maximizing_player=board.turn)
    return best_move