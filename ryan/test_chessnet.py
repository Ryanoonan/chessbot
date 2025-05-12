import pygame
import chess
import chess.engine
import sys
import os
from typing import Optional, Tuple
import torch
import argparse

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the giraffe package
from chessnet import GiraffeEvaluator, get_best_move

MODEL_PATH = "chessnet_untrained_model.pth"

def print_board(board):
    """Print the chess board with coordinates."""
    print("\n   a b c d e f g h")
    print("  -----------------")
    for i in range(8):
        print(f"{8 - i}|", end=" ")
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7 - i))
            print(piece.symbol() if piece else ".", end=" ")
        print(f"|{8 - i}")
    print("  -----------------")
    print("   a b c d e f g h\n")


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


def load_giraffe(model_path=MODEL_PATH):
    """Load the trained Giraffe model."""
    try:
        evaluator = GiraffeEvaluator(model_path)
        print(f"Successfully loaded model from {model_path}")
        return evaluator
    except:
        raise FileNotFoundError(
            f"Model file {model_path} not found. Please train the model first using 'python3 train_giraffe.py'"
        )


def get_human_move(game, board):
    """Get move from human player through pygame interface."""
    game.selected_square = None
    while True:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Only process clicks within the chessboard area
                x, y = pygame.mouse.get_pos()
                if x < game.window_size and y < game.window_size:  # Make sure click is on the board
                    col = x // game.square_size
                    row = 7 - (y // game.square_size)  # Flip row for chess coordinates
                    square = chess.square(col, row)
                    
                    if game.selected_square is None:
                        # First click - select a piece if it's one of the player's pieces
                        piece = board.piece_at(square)
                        if piece and piece.color == chess.WHITE:  # Assuming human is white
                            game.selected_square = square
                    else:
                        # Second click - either move to this square or select a new piece
                        if square == game.selected_square:
                            # Clicked on the same square, deselect it
                            game.selected_square = None
                        else:
                            # Try to make a move
                            move = chess.Move(game.selected_square, square)
                            # Check for promotion
                            if board.piece_at(game.selected_square) and board.piece_at(game.selected_square).piece_type == chess.PAWN:
                                if chess.square_rank(square) == 7:  # Promotion rank for white
                                    move = chess.Move(game.selected_square, square, promotion=chess.QUEEN)
                            
                            if move in board.legal_moves:
                                game.selected_square = None
                                return move
                            else:
                                # Invalid move, try to select a new piece
                                piece = board.piece_at(square)
                                if piece and piece.color == chess.WHITE:
                                    game.selected_square = square
                                else:
                                    game.selected_square = None
        
        # Create list of highlight squares (possible moves from selected square)
        highlight_moves = []
        if game.selected_square is not None:
            for move in board.legal_moves:
                if move.from_square == game.selected_square:
                    highlight_moves.append(move.to_square)
        
        # Update the display
        current_eval = 0
        if hasattr(board, 'current_eval'):
            current_eval = board.current_eval
        game.draw_board(board, current_eval, highlight_moves)
        pygame.time.wait(50)  # Small delay to reduce CPU usage


def play_human_vs_giraffe(model_path):
    """Play chess: Human vs Trained Giraffe."""
    evaluator = load_giraffe(model_path)
    game = ChessGame()
    board = chess.Board()

    # Get initial evaluation
    initial_eval = evaluator.evaluate_position(board)
    game.draw_board(board, initial_eval)

    while not board.is_game_over():
        current_eval = evaluator.evaluate_position(board, True)
        print("Current evaluation: ", current_eval)
        game.draw_board(board, current_eval)
        # Human's turn (White)
        move = get_human_move(game, board)
        if move is None:  # Game quit
            break

        board.push(move)
        current_eval = evaluator.evaluate_position(board, True)
        print("Current evaluation: ", current_eval)
        game.draw_board(board, current_eval)

        if board.is_game_over():
            break

        # Giraffe's turn (Black)
        print("Giraffe is thinking...")
        move = get_best_move(board, depth=3, evaluator=evaluator)
        print(f"Giraffe plays: {move}")
        board.push(move)
        

    print("Game Over!")
    print("Result:", board.result())


def giraffe_vs_stockfish(num_games=5, model_path=MODEL_PATH):
    """Pit the trained Giraffe against Stockfish."""
    evaluator = load_giraffe(model_path)
    engine = chess.engine.SimpleEngine.popen_uci(
        "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"
    )
    engine.configure({"Skill Level": 0})  # Adjust Stockfish skill level as desired

    results = {"giraffe_wins": 0, "stockfish_wins": 0, "draws": 0}
    game = ChessGame()  # For visualization
    
    # Set up window title
    pygame.display.set_caption("Giraffe vs Stockfish")

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        board = chess.Board()
        current_eval = evaluator.evaluate_position(board)
        game.draw_board(board, current_eval)
        # Allow user to see initial position
        pygame.time.wait(1000)
        
        # Check for window close between games
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                engine.quit()
                return
        move_count = 0
        while not board.is_game_over():

            # Process any window events to keep responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    engine.quit()
                    return
                    
            move_count += 1
            
            # Giraffe's turn (White)
            print(f"Move {move_count}: Giraffe (White) is thinking...")
            move = get_best_move(board, depth=3, evaluator=evaluator)
            board.push(move)
            current_eval = evaluator.evaluate_position(board)
            print(f"Giraffe played: {move}")
            game.draw_board(board, current_eval)
            
            # Pause to show the move
            pygame.time.wait(80)
            
            # Check for window close or game over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    engine.quit()
                    return
                    
            if board.is_game_over():
                break

            move_count += 1
                
            # Stockfish's turn (Black)
            print(f"Move {move_count}: Stockfish (Black) is thinking...")
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            current_eval = evaluator.evaluate_position(board)
            print(f"Stockfish played: {result.move}")
            game.draw_board(board, current_eval)
            
            # Pause to show the move
            pygame.time.wait(800)

        # Show final position for longer
        pygame.time.wait(2000)
        
        result = board.result()
        print(f"Game {game_num + 1} Result: {result}")

        if result == "1-0":
            results["giraffe_wins"] += 1
            print("Giraffe wins!")
        elif result == "0-1":
            results["stockfish_wins"] += 1
            print("Stockfish wins!")
        else:
            results["draws"] += 1
            print("Game drawn!")
            
        # Wait a moment before starting next game
        pygame.time.wait(1500)

    print("\nFinal Results:")
    print(f"Giraffe Wins: {results['giraffe_wins']}")
    print(f"Stockfish Wins: {results['stockfish_wins']}")
    print(f"Draws: {results['draws']}")
    
    # Keep the window open until user closes it
    print("Close the window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    engine.quit()
    pygame.quit()


def model_vs_model(num_games=5, model1_path=MODEL_PATH, model2_path=MODEL_PATH):
    """Pit two different Giraffe models against each other."""
    evaluator1 = load_giraffe(model1_path)
    evaluator2 = load_giraffe(model2_path)
    
    # Extract model names for display
    model1_name = os.path.basename(model1_path).replace('.pth', '')
    model2_name = os.path.basename(model2_path).replace('.pth', '')
    
    results = {"model1_wins": 0, "model2_wins": 0, "draws": 0}
    game = ChessGame()  # For visualization
    
    # Set up window title
    pygame.display.set_caption(f"{model1_name} vs {model2_name}")

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        board = chess.Board()
        # Use evaluator1 for initial evaluation
        current_eval = evaluator1.evaluate_position(board)
        game.draw_board(board, current_eval)
        
        # Allow user to see initial position
        pygame.time.wait(1000)
        # Check for window close between games
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        move_count = 0
        while not board.is_game_over():
            # Process any window events to keep responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            move_count += 1
            
            # Model 1's turn (White)
            print(f"Move {move_count}: {model1_name} (White) is thinking...")
            move = get_best_move(board, depth=3, evaluator=evaluator1)
            board.push(move)
            current_eval = evaluator1.evaluate_position(board)
            print(f"{model1_name} played: {move}")
            game.draw_board(board, current_eval)
            
            # Pause to show the move
            pygame.time.wait(800)
            
            # Check for window close or game over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            if board.is_game_over():
                break

            move_count += 1
                
            # Model 2's turn (Black)
            print(f"Move {move_count}: {model2_name} (Black) is thinking...")
            move = get_best_move(board, depth=3, evaluator=evaluator2)
            board.push(move)
            # Use evaluator1 for consistent evaluation display
            current_eval = evaluator1.evaluate_position(board)
            print(f"{model2_name} played: {move}")
            game.draw_board(board, current_eval)
            
            # Pause to show the move
            pygame.time.wait(800)

        # Show final position for longer
        pygame.time.wait(2000)
        
        result = board.result()
        print(f"Game {game_num + 1} Result: {result}")

        if result == "1-0":
            results["model1_wins"] += 1
            print(f"{model1_name} wins!")
        elif result == "0-1":
            results["model2_wins"] += 1
            print(f"{model2_name} wins!")
        else:
            results["draws"] += 1
            print("Game drawn!")
            
        # Wait a moment before starting next game
        pygame.time.wait(1500)

    print("\nFinal Results:")
    print(f"{model1_name} Wins: {results['model1_wins']}")
    print(f"{model2_name} Wins: {results['model2_wins']}")
    print(f"Draws: {results['draws']}")
    
    # Keep the window open until user closes it
    print("Close the window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chess: Play against trained Giraffe or watch it play against AI opponents"
    )
    parser.add_argument(
        "--mode",
        choices=["human", "stockfish", "model_vs_model"],
        default="human",
        help="Play mode: human (play against Giraffe), stockfish (watch Giraffe vs Stockfish), or model_vs_model (pit two Giraffe models against each other)",
    )
    parser.add_argument(
        "--games", type=int, default=5, help="Number of games to play in AI vs AI modes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the model file to load (for human and stockfish modes)",
    )
    parser.add_argument(
        "--model1",
        type=str,
        default=MODEL_PATH,
        help="Path to the first model file (White) for model_vs_model mode",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=MODEL_PATH,
        help="Path to the second model file (Black) for model_vs_model mode",
    )

    args = parser.parse_args()

    if args.mode == "human":
        play_human_vs_giraffe(args.model)
    elif args.mode == "stockfish":
        giraffe_vs_stockfish(args.games, args.model)
    elif args.mode == "model_vs_model":
        model_vs_model(args.games, args.model1, args.model2)
