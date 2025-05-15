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