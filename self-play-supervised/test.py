import chess
import chess.engine
import pygame
import torch
from helper import ChessGame, get_best_move
from chessnet import ChessNet
import argparse
from helper import board_to_tensor


def giraffe_vs_stockfish(model_path, num_games=5):
    """Pit the trained Giraffe against Stockfish."""
    evaluator = ChessNet()
    evaluator.load_state_dict(torch.load(model_path))
    evaluator.eval()  # Set model to evaluation mode
    engine = chess.engine.SimpleEngine.popen_uci(
        "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"
    )
    engine.configure({"Skill Level": 5})  # Adjust Stockfish skill level as desired

    results = {"giraffe_wins": 0, "stockfish_wins": 0, "draws": 0}
    game = ChessGame()  # For visualization
    
    # Set up window title
    pygame.display.set_caption("Giraffe vs Stockfish")

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        board = chess.Board()
        current_eval = evaluator(board_to_tensor(board)).item()
        print(current_eval)
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
            move = get_best_move(board, depth=2, evaluator=evaluator)
            board.push(move)
            current_eval = evaluator(board_to_tensor(board)).item()
            # Print the move and evaluation
            print(f"Giraffe played: {move}, Evaluation: {current_eval}")
            game.draw_board(board, current_eval)
            
            # Pause to show the move (Edit this for longer pause)
            pygame.time.wait(1)
            
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
            result = engine.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
            current_eval = evaluator(board_to_tensor(board)).item()
            print(f"Stockfish played: {result.move}, Our Evaluation: {current_eval}")
            # Print the evaluation
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

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Giraffe vs Stockfish chess matches.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained Giraffe model.")
    args = parser.parse_args()

    model_path = args.model

    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((800, 800))
    
    # Run the Giraffe vs Stockfish match
    giraffe_vs_stockfish(num_games=1, model_path=model_path)
    
    # Quit Pygame
    pygame.quit()