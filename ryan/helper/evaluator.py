import chess
import torch

from helper.see import static_exchange_evaluation
from helper.fast_eval_pos import FastChessEvaluatePosition


class Evaluator: 
    def get_best_move(self, board):
        pass


class MinimaxAlphaBetaEvaluator(Evaluator):
    def __init__(self, depth=3):
        self.depth = depth

    def get_best_move(self, board):
        # Perform minimax search with alpha-beta pruning
        _ , best_move = self.minimax_search(board, alpha=float("-inf"), depth = self.depth, beta=float("inf"))
        return best_move
        

    def evaluate_position(self, board):
        # Count the material in the position
        pass
        

    def minimax_search(
        self,
        board: chess.Board,
        depth: int = 3,
        maximizing_player: bool = True,
        alpha: float = float("-inf"),
        beta: float = float("inf")
    ) -> tuple[float, chess.Move | None]:
        
        moves = list(board.legal_moves)
        # Simple move ordering: prioritize captures and promotions
        moves.sort(key=lambda move: (
        (1000 if board.is_capture(move) else 0) + 
        (1500 if move.promotion else 0) +
        (800 if board.gives_check(move) else 0)
        ), reverse=True)

        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta, maximizing_player), None

        if maximizing_player:
            max_eval = float("-inf")
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax_search(board, depth - 1, False, alpha, beta)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval, best_move
        else:
            min_eval = float("inf")
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax_search(board, depth - 1, True, alpha, beta)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval, best_move
        
    def quiescence_search(
    self,
    board: chess.Board,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    depth: int = 0,
    max_depth: int = 4
) -> float:
        # 1) Stand‐pat
        stand_pat = self.evaluate_position(board)
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        # 2) Only “good” captures (SEE ≥ 0) and promotions
        for move in board.legal_moves:
            if board.is_capture(move):
                # static exchange eval: negative means material loss
                if static_exchange_evaluation(board, move) < 0:
                    continue
            elif not move.promotion:
                continue

            board.push(move)
            score = self.quiescence_search(
                board,
                alpha, beta,
                not maximizing_player,
                depth=depth + 1,
            )
            board.pop()

            if maximizing_player:
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            else:
                if score <= alpha:
                    return alpha
                beta = min(beta, score)

        return alpha if maximizing_player else beta
        
class MaterialMinimax(MinimaxAlphaBetaEvaluator):
    def __init__(self, depth=3):
        super().__init__(depth)
        self.material_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }

    def evaluate_position(self, board):
        # Count the material in the position
        material_score = 0
        for piece_type in self.material_values:
            material_score += len(board.pieces(piece_type, chess.WHITE)) * self.material_values[piece_type]
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * self.material_values[piece_type]
        return material_score
    
class NeuralNetworkEvaluator(MinimaxAlphaBetaEvaluator):
    def __init__(self, model, board_to_tensor, depth=3):
        super().__init__(depth)
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.board_to_tensor = board_to_tensor

    def get_best_move(self, board):
        # No need to rebuild accumulator - using direct forward pass
        return super().get_best_move(board)

    def evaluate_position(self, board):
        # Use the model directly with the board_to_tensor function
        with torch.no_grad():  # No need to track gradients for inference
            tensor = self.board_to_tensor(board)
            tensor = tensor.unsqueeze(0) if tensor.dim() == 4 else tensor  # Add batch dimension if needed
            evaluation = self.model(tensor)
            return evaluation.item()  # Convert from tensor to scalar