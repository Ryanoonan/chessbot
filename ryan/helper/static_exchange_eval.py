# Simple piece‐value table (in pawns)
import chess


PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}

def simple_see(board: chess.Board, move: chess.Move) -> int:
    """
    Approximate Static Exchange Evaluation for a single capture.
    Returns the net material gain (positive if good for side to move).
    """
    if not board.is_capture(move):
        return 0
    # Value of the captured piece
    victim = board.piece_at(move.to_square)
    if victim is None:
        return -100
    
    gain = PIECE_VALUES[victim.piece_type]
    # Make a copy and play the capture
    board_copy = board.copy(stack=False)
    board_copy.push(move)
    # Find all legal recaptures on the same square
    recaptures = [
        m for m in board_copy.legal_moves
        if board_copy.is_capture(m) and m.to_square == move.to_square
    ]
    if not recaptures:
        return gain
    # Pick the least‐valuable attacker for the next recapture
    next_move = min(
        recaptures,
        key=lambda m: PIECE_VALUES[board_copy.piece_at(m.from_square).piece_type]
    )
    # Subtract the opponent’s reply recursively
    return gain - simple_see(board_copy, next_move)

def is_quiet(board: chess.Board, threshold: int = 0) -> bool:
    """
    Returns True iff there are no captures with SEE ≤ threshold,
    no checks, and no promotions—i.e. a “quiet” position.
    """
    for move in board.legal_moves:
        if board.is_capture(move) and simple_see(board, move) <= threshold:
            return False
        if board.gives_check(move) or move.promotion is not None:
            return False
    return True