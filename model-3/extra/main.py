from chess_ai import ChessAI

# Initialize the AI with the trained model
ai = ChessAI(model_path="chess_eval_model.pth")

# Play a game against the AI
while not ai.get_game_state():
    best_move = ai.get_best_move(depth=3)  # You can adjust the depth as needed
    print(f"AI Move: {best_move}")
    ai.make_move(best_move)
    print(ai.display_board())
