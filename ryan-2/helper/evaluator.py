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

    def get_best_moves(self, board):
        # Perform minimax search with alpha-beta pruning
        _ , best_moves = self.minimax_search(board, alpha=float("-inf"), depth = self.depth, beta=float("inf"))
        return best_moves
        

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
    ) -> tuple[float, list[chess.Move]]:
        
        moves = list(board.legal_moves)
        # Simple move ordering: prioritize captures and promotions
        moves.sort(key=lambda move: (
        (1000 if board.is_capture(move) else 0) + 
        (1500 if move.promotion else 0) +
        (800 if board.gives_check(move) else 0)
        ), reverse=True)

        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta, maximizing_player), []

        if maximizing_player:
            max_eval = float("-inf")
            best_moves = []

            for move in moves:
                board.push(move)
                eval, _ = self.minimax_search(board, depth - 1, False, alpha, beta)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_moves = [move]
                elif eval == max_eval:
                    best_moves.append(move)
                
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval, best_moves
        else:
            min_eval = float("inf")
            best_moves = []

            for move in moves:
                board.push(move)
                eval, _ = self.minimax_search(board, depth - 1, True, alpha, beta)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_moves = [move]
                elif eval == min_eval:
                    best_moves.append(move)
                
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval, best_moves
        
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
        if board.is_checkmate():
            return float("-inf") if board.turn == chess.WHITE else float("inf")

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
        

class HybridMaterialNeural(MinimaxAlphaBetaEvaluator):
    def __init__(self, model, board_to_tensor, depth=2):
        super().__init__(depth)
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.board_to_tensor = board_to_tensor
        self.material_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
    
    def evaluate_position(self, board):
        """Only used during minimax search - using material only"""
        return self.material_evaluation(board)

    def material_evaluation(self, board):
        """Pure material evaluation"""
        if board.is_checkmate():
            return float("-inf") if board.turn == chess.WHITE else float("inf")

        material_score = 0
        for piece_type in self.material_values:
            material_score += len(board.pieces(piece_type, chess.WHITE)) * self.material_values[piece_type]
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * self.material_values[piece_type]
        
        return material_score

    def network_evaluation(self, board):
        """Pure neural network evaluation"""
        with torch.no_grad():
            tensor = self.board_to_tensor(board)
            tensor = tensor.unsqueeze(0) if tensor.dim() == 4 else tensor
            neural_score = self.model(tensor).item()
        
        return neural_score
    
    def get_best_move(self, board):
        """Two-stage evaluation: 
        1. Run minimax with material evaluation to get top material moves
        2. Choose the best move among these using neural evaluation
        """
        # First run minimax with material evaluation
        _, material_best_moves = self.minimax_search(board, alpha=float("-inf"), depth=self.depth, beta=float("inf"))
        
        if not material_best_moves:
            return None
            
        print(f"Top material moves with equal evaluation ({len(material_best_moves)}):")
        for move in material_best_moves:
            print(f"- {move}")
            
        # If only one move, no need for neural evaluation
        if len(material_best_moves) == 1:
            return material_best_moves[0]
            
        # Use neural network to evaluate the top material moves
        best_move = None
        best_score = float("-inf")
        
        for move in material_best_moves:
            board.push(move)
            neural_score = self.network_evaluation(board)
            board.pop()
            
            if neural_score > best_score:
                best_score = neural_score
                best_move = move
                
        print(f"Best move selected by neural evaluation: {best_move}")
        return best_move