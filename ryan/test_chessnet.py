import pygame
import chess
import chess.engine
import sys
import os
from typing import Optional, Tuple
import torch
import argparse
from chess_game import ChessGame
from helper.evaluator import MaterialMinimax, NeuralNetworkEvaluator
from helper.board_to_tensor import board_to_tensor_nnue

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the giraffe package
from chessnet import ChessNet, GiraffeEvaluator, get_best_move

MODEL_PATH = "chessnet_untrained_model.pth"





def load_model(model_path=MODEL_PATH):
    """Load the trained Giraffe model."""
    try:
        evaluator = GiraffeEvaluator(model_path)
        print(f"Successfully loaded model from {model_path}")
        return evaluator
    except:
        raise FileNotFoundError(
            f"Model file {model_path} not found. Please train the model first using 'python3 train_giraffe.py'"
        )



def model_vs_stockfish(num_games=5, model_path=MODEL_PATH, evaluator=None):
    """Pit the trained Giraffe against Stockfish."""
    if evaluator is None:
        model = ChessNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        evaluator = NeuralNetworkEvaluator(model, depth = 3, board_to_tensor = board_to_tensor_nnue)
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
            move = evaluator.get_best_move(board)
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
            pygame.time.wait(80)

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chess: Watch Giraffe play against Stockfish"
    )
    parser.add_argument(
        "--games", type=int, default=5, help="Number of games to play"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the Giraffe model file to load",
    )

    parser.add_argument(
        "--material-minimax",
        type=bool,
        default=False,
        help="Use material minimax evaluation instead of a Neural Network",
    )

    args = parser.parse_args()
    evaluator = None
    if args.material_minimax:
        evaluator = MaterialMinimax(depth=4)
    model_vs_stockfish(args.games, args.model, evaluator)
