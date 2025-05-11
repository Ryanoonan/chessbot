import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
import numpy as np
import random
from tqdm import tqdm
import os
from dotenv import load_dotenv
from chessnet import ChessNet
from helper import board_to_tensor

# Load environment variables
load_dotenv()

# Set up Stockfish path
STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if STOCKFISH_PATH is None:
    STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"


def get_stockfish_evaluation(board: chess.Board, engine, time_limit=0.1) -> float:
    """Get Stockfish evaluation for a position."""
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = result["score"].white().score(mate_score=10000)
    return score / 100.0  # Convert to pawns (from centipawns)

def self_play_train(model, num_games=100, num_epochs=10, learning_rate=0.001, epsilon=0.1):
    """Train ChessNet through self-play with epsilon-greedy exploration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Initialize Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_positions = 0
        
        for game in tqdm(range(num_games), desc=f"Self-play epoch {epoch+1}/{num_epochs}"):
            board = chess.Board()
            game_history = []
            
            # Play until game is over or max moves reached
            max_moves = 100
            move_count = 0
            
            while not board.is_game_over() and move_count < max_moves:
                # Get all legal moves
                legal_moves = list(board.legal_moves)
                
                # Epsilon-greedy move selection
                if random.random() < epsilon:
                    # Random move
                    move = random.choice(legal_moves)
                else:
                    # Use neural network to evaluate all legal moves
                    move_evaluations = []
                    for move in legal_moves:
                        # Make the move on a copy of the board
                        board_copy = board.copy()
                        board_copy.push(move)
                        
                        # Convert board to tensor
                        tensor = board_to_tensor(board_copy).to(device)
                        
                        # Evaluate position with neural network
                        with torch.no_grad():
                            evaluation = model(tensor.unsqueeze(0)).item()
                            
                        # Negate evaluation for black (since model gives evaluation from white's perspective)
                        if not board.turn:  # If it's black's turn
                            evaluation = -evaluation
                            
                        move_evaluations.append((move, evaluation))
                    
                    # Choose move with highest evaluation
                    move_evaluations.sort(key=lambda x: x[1], reverse=True)
                    move = move_evaluations[0][0]
                
                # Record position before making the move
                current_tensor = board_to_tensor(board).to(device)
                
                # Make the move
                board.push(move)
                move_count += 1
                
                # Get Stockfish evaluation of the resulting position
                stockfish_eval = get_stockfish_evaluation(board, engine, time_limit=0.01)
                
                # Adjust evaluation based on whose turn it was
                if not board.turn:  # If black just moved, negate the evaluation
                    stockfish_eval = -stockfish_eval
                
                # Save the position and evaluation for training
                game_history.append((current_tensor, stockfish_eval))
                
                # If we have enough positions, perform a training step
                if len(game_history) >= 32:
                    # Create batch
                    positions, evaluations = zip(*game_history)
                    position_batch = torch.stack(positions)
                    evaluation_batch = torch.tensor(evaluations, dtype=torch.float32, device=device).unsqueeze(1)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(position_batch)
                    loss = criterion(outputs, evaluation_batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_positions += len(game_history)
                    
                    # Clear game history after training
                    game_history = []
            
            # Train on any remaining positions from the game
            if game_history:
                positions, evaluations = zip(*game_history)
                position_batch = torch.stack(positions)
                evaluation_batch = torch.tensor(evaluations, dtype=torch.float32, device=device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(position_batch)
                loss = criterion(outputs, evaluation_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_positions += len(game_history)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / max(1, num_positions)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save model after each epoch
        torch.save(model.state_dict(), f"chessnet_self_play_epoch_{epoch+1}.pth")
    
    # Close the Stockfish engine
    engine.quit()
    
    # Save final model
    torch.save(model.state_dict(), "chessnet_self_play_final.pth")
    
    return model

def main():
    # Initialize model
    model = ChessNet()
    
    # Train model with self-play
    trained_model = self_play_train(
        model,
        num_games=50,       # Number of self-play games per epoch
        num_epochs=20,      # Number of training epochs
        learning_rate=0.001,
        epsilon=0.1         # Exploration rate for epsilon-greedy strategy
    )
    
    print("Self-play training completed!")

if __name__ == "__main__":
    main()