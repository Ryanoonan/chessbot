import chess
import torch


def board_to_tensor_1(board: chess.Board) -> torch.Tensor:
    """Convert a chess board to a tensor representation using 12 feature planes.
    
    Returns a tensor of shape (12, 8, 8) where each plane represents a different piece type:
    - Planes 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    - Planes 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    
    Each cell contains 1 if the piece is present, 0 otherwise.
    """
    # Initialize tensor with 12 planes (6 piece types × 2 colors)
    tensor = torch.zeros(12, 8, 8)
    
    # Maps piece symbol to plane index
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 1
    }
    
    # Fill tensor with piece locations
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get piece type and color
            piece_type = piece.piece_type
            color = piece.color
            
            # Calculate plane index (0-5 for white, 6-11 for black)
            plane_idx = piece_to_plane[piece_type]
            if not color:  # If black
                plane_idx += 6
                
            # Set value to 1 at the piece's position
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[plane_idx, rank, file] = 1.0

    
    return tensor


def board_to_tensor_nnue(board: chess.Board) -> torch.Tensor:
    piece_type_to_plane = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
    }
    # dims: [king_square, piece_square, plane (0–9), perspective (0=white,1=black)]
    X = torch.zeros(64, 64, 10, 2, dtype=torch.float32)

    # Gather king squares by color
    white_king_sq = [sq for sq in chess.SQUARES
                     if (p := board.piece_at(sq)) and p.piece_type == chess.KING and p.color == chess.WHITE]
    black_king_sq = [sq for sq in chess.SQUARES
                     if (p := board.piece_at(sq)) and p.piece_type == chess.KING and p.color == chess.BLACK]

    # For each king perspective
    for perspective, king_list in enumerate((white_king_sq, black_king_sq)):
        # perspective 0 = white-king view; 1 = black-king view
        for king_sq in king_list:
            # For each piece on the board
            for piece_sq in chess.SQUARES:
                if (bp := board.piece_at(piece_sq)) is None:
                    continue
                # Skip if piece is a king (exclude kings from piece_square dimension)
                if bp.piece_type == chess.KING:
                    continue
                base_plane = piece_type_to_plane[bp.piece_type]
                color_offset = 0 if bp.color == chess.WHITE else 5
                plane = base_plane + color_offset
                X[king_sq, piece_sq, plane, perspective] = 1.0

    return X
