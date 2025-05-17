import chess
import math

def minimax_search(board, depth, model, alpha=-math.inf, beta=math.inf, maximizing_player=True):
    """Performs the minimax search with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        return model.eval_position(board).item()  # Return the evaluation of the board

    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_search(board, depth-1, model, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move
    else:
        min_eval = math.inf
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_search(board, depth-1, model, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move
