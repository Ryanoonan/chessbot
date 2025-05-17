import torch
import chess
from stockfish import Stockfish
from chess_model import ChessEvalNN  # Import the ChessEvalNN model

# Load the trained model
model = ChessEvalNN()
model.load_state_dict(torch.load("chess_eval_model.pth"), strict= False)
model.eval()

# Stockfish Setup
stockfish = Stockfish(path="/Users/mahnoorabbas/chess-ai/stockfish/stockfish-macos-m1-apple-silicon")
stockfish.set_depth(20)
stockfish.set_skill_level(20)

# Convert Board to Model Input (Tensor)
def board_to_tensor(board):
    tensor = torch.zeros((12, 8, 8), dtype=torch.float)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        idx = "PNBRQKpnbrqk".index(piece.symbol())
        row, col = divmod(square, 8)
        tensor[idx][row][col] = 1
    return tensor.view(1, -1)

# Select Best Move Using Your Model
def select_best_move(board, model):
    best_score = -float('inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        input_tensor = board_to_tensor(board)
        with torch.no_grad():
            score = model(input_tensor).item()
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# Game Setup
board = chess.Board()

# Print initial board state
print("Initial Board:")
print(board)

# Game Loop (Alternating moves between the model and Stockfish)
while not board.is_game_over():
    # Model's Move
    model_move = select_best_move(board, model)
    print(f"Model's Move: {model_move}")
    board.push(model_move)
    print(board)

    if board.is_game_over():
        break

    # Stockfish's Move
    stockfish.set_fen_position(board.fen())
    stockfish_move = stockfish.get_best_move()
    print(f"Stockfish's Move: {stockfish_move}")
    board.push_uci(stockfish_move)
    print(board)

# End of Game
if board.is_checkmate():
    print("Checkmate!")
elif board.is_stalemate():
    print("Stalemate!")
elif board.is_insufficient_material():
    print("Insufficient Material!")
elif board.is_seventyfive_moves():
    print("75-move Rule Draw!")
elif board.is_variant_draw():
    print("Variant Draw!")
else:
    print("Game Over!")
