import torch
import torch.nn as nn
import chess
from stockfish import Stockfish

# Your model
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(12*8*8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Board Setup
board = chess.Board()
print("Legal moves:", board.legal_moves)
print("Checkmate:", board.is_checkmate())
board.push_san('e4')  # Example move
print(board)

# Stockfish Setup
stockfish = Stockfish(path="/Users/mahnoorabbas/chess-ai/stockfish/stockfish-macos-m1-apple-silicon")
stockfish.set_depth(0)
stockfish.set_skill_level(0)
print("Parameters:", stockfish.get_parameters())
stockfish.set_fen_position(board.fen())
stockfish_eval = stockfish.get_evaluation()
print("Stockfish Evaluation:", stockfish_eval)

# Your trained model
model = YourModel()
model.load_state_dict(torch.load("chess_eval_model.pth"), strict=False)
model.eval()

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

# Model's Move
model_move = select_best_move(board, model)
print("Model's Best Move:", model_move)

# Stockfish's Best Move
stockfish_move = stockfish.get_best_move()
print("Stockfish's Best Move:", stockfish_move)

# Compare the Moves
if model_move.uci() == stockfish_move:
    print("Model and Stockfish agree on the move!")
else:
    print("Model and Stockfish disagree on the move.")
