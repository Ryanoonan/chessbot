import torch
import torch.nn as nn
import chess
from typing import Tuple, Optional, List
import math
from chessnet import GiraffeEvaluator
#
class MCTSNode:
    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"] = None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.41):
        # UCB1 formula
        def ucb_score(child):
            if child.visits == 0:
                return float("inf")
            exploitation = child.value_sum / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)  # negate because it's a 2-player game

    def is_terminal(self):
        return self.board.is_game_over()

    def get_result(self, evaluator):
        # Use neural network evaluator
        return evaluator.evaluate_position(self.board)


def mcts(board: chess.Board, evaluator, num_simulations: int = 1000, c_param: float = 1.41) -> Optional[chess.Move]:
    root = MCTSNode(board)

    for _ in range(num_simulations):
        node = root

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(c_param)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # Simulation
        value = node.get_result(evaluator)

        # Backpropagation
        node.backpropagate(value)

    # Choose the most visited move
    best_move_node = max(root.children, key=lambda n: n.visits, default=None)
    return best_move_node.move if best_move_node else None

def quiescence_search(board: chess.Board, evaluator: GiraffeEvaluator, alpha: float = float("-inf"), beta: float = float("inf")) -> float:
    stand_pat = evaluator.evaluate_position(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue  # Only search noisy moves (captures in this case)

        board.push(move)
        score = -quiescence_search(board, evaluator, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def modified_minimax_search(
    board: chess.Board,
    depth: int,
    evaluator: GiraffeEvaluator,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing_player: bool = True,
) -> Tuple[float, Optional[chess.Move]]:
    """Minimax search with alpha-beta pruning and quiescence search."""
    
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, evaluator, alpha, beta), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = modified_minimax_search(board, depth - 1, evaluator, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = modified_minimax_search(board, depth - 1, evaluator, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_move

def modified_get_best_move(
    board: chess.Board, depth: int, evaluator: GiraffeEvaluator
) -> chess.Move:
    """Get the best move for the current position using minimax search."""
    _, best_move = modified_minimax_search(board, depth, evaluator, maximizing_player=board.turn)
    return best_move