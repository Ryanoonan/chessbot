from stockfish import Stockfish
import chess
import chess.svg


#Board
board = chess.Board()
print("Legal moves:",board.legal_moves )
print("Checkmate:", board.is_checkmate())
board.push_san('e4')
print(board)
print("Fen:", board.fen())


# Stockfish
stockfish = Stockfish(path="/Users/mahnoorabbas/chess-ai/stockfish/stockfish-macos-m1-apple-silicon")
stockfish.set_depth(20)
stockfish.set_skill_level(20)
print("Parameters:", stockfish.get_parameters())
stockfish.set_fen_position(board.fen())
print("Eval" ,stockfish.get_evaluation())

# # Set a position on the board (start position)
# stockfish.set_position(["startpos"])

# # Get the best move from the current position
# best_move = stockfish.get_best_move()

# # Print the best move to see if Stockfish is working
# print("Best move:", best_move)
