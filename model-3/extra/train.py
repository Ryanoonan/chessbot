import chess
import chess.engine
from neural_network import ChessEvalModel
import torch

def generate_training_data(num_games=100):
    """Generate training data using Stockfish."""
    stockfish = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")
    data = []

    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            result = stockfish.analyse(board, chess.engine.Limit(time=0.1))
            evaluation = result["score"].relative.score(mate_score=10000)  # Get evaluation in centipawns
            data.append((board.copy(), evaluation))
            move = stockfish.play(board, chess.engine.Limit(time=0.1)).move
            board.push(move)
    
    stockfish.quit()
    return data

def train_model():
    model = ChessEvalModel()
    data = generate_training_data()

    # Train the model on (board, eval) pairs
    model.train(data, epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), "chess_eval_model.pth")

if __name__ == "__main__":
    train_model()
