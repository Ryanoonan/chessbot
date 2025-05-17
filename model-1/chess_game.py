import os
import chess
import pygame


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