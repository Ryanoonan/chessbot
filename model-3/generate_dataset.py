# generate_dataset.py

import chess
import chess.engine
import random
import csv

# Path to your Stockfish binary
STOCKFISH_PATH = "/Users/mahnoorabbas/chess-ai/stockfish/stockfish-macos-m1-apple-silicon"

# Output file
OUTPUT_CSV = "chess_dataset.csv"

# Number of positions to generate
NUM_SAMPLES = 5000  # Start small to test; increase later

def generate_random_board(max_moves=20):
    board = chess.Board()
    moves = random.randint(5, max_moves)
    for _ in range(moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board

def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fen", "score_cp"])

        for i in range(NUM_SAMPLES):
            board = generate_random_board()
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                score = info["score"].white().score(mate_score=10000)
                if score is not None:
                    writer.writerow([board.fen(), score])
            except Exception as e:
                print(f"Error on sample {i}: {e}")

            if i % 50 == 0:
                print(f"{i} positions processed")

    engine.quit()
    print(f"Done! Dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
