import chess
import torch

def encode_features(board: chess.Board):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    feat_w, feat_b = [], []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            idx = (p.piece_type - 1) + (0 if p.color else 5)  # 0–9
            feat_w.append(idx * 64 + (sq ^ wk))
            feat_b.append(idx * 64 + (sq ^ bk))
    return feat_w, feat_b

class FastChessEvaluatePosition:
    def __init__(self, model):
        # extract hidden‐layer tables
        self.W1w = model.fc1w.weight.data.t().clone()  # [in_features × hidden]
        self.b1w = model.fc1w.bias.data.clone()        # [hidden]
        self.W1b = model.fc1b.weight.data.t().clone()
        self.b1b = model.fc1b.bias.data.clone()
        # extract downstream layers
        self.W2 = model.fc2.weight.data.clone()
        self.b2 = model.fc2.bias.data.clone()
        self.W3 = model.fc3.weight.data.clone()
        self.b3 = model.fc3.bias.data.clone()
        self.W4 = model.fc4.weight.data.clone()
        self.b4 = model.fc4.bias.data.clone()
        self.acc_w = None
        self.acc_b = None

    def rebuild(self, board: chess.Board):
        fw, fb = encode_features(board)
        self.acc_w = self.b1w.clone()
        for f in fw:
            self.acc_w += self.W1w[:, f]
        self.acc_b = self.b1b.clone()
        for f in fb:
            self.acc_b += self.W1b[:, f]

    def update(self, board: chess.Board, move: chess.Move):
        bc = board.copy()
        fw0, fb0 = encode_features(bc)
        king_moved = bc.piece_at(move.from_square).piece_type == chess.KING
        bc.push(move)
        fw1, fb1 = encode_features(bc)
        if king_moved:
            self.rebuild(board)
        else:
            rm_w = set(fw0) - set(fw1)
            ad_w = set(fw1) - set(fw0)
            rm_b = set(fb0) - set(fb1)
            ad_b = set(fb1) - set(fb0)
            for f in rm_w: self.acc_w -= self.W1w[:, f]
            for f in ad_w: self.acc_w += self.W1w[:, f]
            for f in rm_b: self.acc_b -= self.W1b[:, f]
            for f in ad_b: self.acc_b += self.W1b[:, f]

    def evaluate_position(self):
        h_w = torch.relu(self.acc_w)
        h_b = torch.relu(self.acc_b)
        h   = torch.cat([h_w, h_b], dim=0)
        z2  = torch.relu(self.W2 @ h + self.b2)
        z3  = torch.relu(self.W3 @ z2 + self.b3)
        out = (self.W4 @ z3 + self.b4).item()
        return out

