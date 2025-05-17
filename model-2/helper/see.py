import chess

# piece values in centipawns
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:  20000,  # add this line
}

def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Return SEE >= 0 if the capture sequence on move.to_square
    is material‐winning for the side to move, < 0 if material‐losing.
    """
    if not board.is_capture(move):
        return 0
    to_sq = move.to_square
    # clone so we don’t touch the real board
    b = board.copy()
    # value of the piece that’s about to be captured
    captured = b.piece_at(to_sq)
    if not captured:
        return 0
    gains = [PIECE_VALUES[captured.piece_type]]
    b.push(move)

    # build list of successive least‐valuable attackers
    side = b.turn
    while True:
        attackers = b.attackers(side, to_sq)
        if not attackers:
            break
        # pick the cheapest attacker
        from_sq = min(
            attackers,
            key=lambda sq: PIECE_VALUES[b.piece_at(sq).piece_type]
        )
        attacker = b.piece_at(from_sq)
        gains.append(PIECE_VALUES[attacker.piece_type])
        # simulate that capture
        b.push(chess.Move(from_sq, to_sq))
        side = b.turn

    # now do the “swap‐off” minimax from the back
    for i in range(len(gains) - 2, -1, -1):
        gains[i] = max(gains[i], -gains[i+1])

    return gains[0]
