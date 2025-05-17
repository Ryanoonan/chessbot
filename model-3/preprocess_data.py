import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import chess
from sklearn.preprocessing import StandardScaler

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = []
        self.y = []

        for _, row in self.data.iterrows():
            board = chess.Board(row['fen'])
            encoded = self.encode_board(board)
            self.X.append(encoded)

            y_val = float(row['score_cp'])/100
            if y_val > 10 or y_:
                y_val = 10
            elif y_val < -10:
                y_val = -10
            self.y.append(y_val)
        

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def encode_board(self, board):
        piece_map = board.piece_map()
        planes = np.zeros((12, 8, 8), dtype=np.float32)

        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color_offset = 0 if piece.color == chess.WHITE else 6
            row = 7 - (square // 8)
            col = square % 8
            planes[piece_type + color_offset][row][col] = 1

        return planes.flatten()

# Example usage
if __name__ == "__main__":
    dataset = ChessDataset("chess_dataset.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Loaded {len(dataset)} samples.")
    for xb, yb in dataloader:
        print("Batch X shape:", xb.shape)
        print("Batch y shape:", yb.shape)
        break
