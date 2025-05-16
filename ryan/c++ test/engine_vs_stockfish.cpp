// engine_vs_stockfish.cpp
#include "engine_vs_stockfish.h"

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include "board.h"
#include "minimax.h"
#include "uci_engine.h"
#include "chessnet_evaluator.h"
#include "fen_utils.h"
#include "evaluator.h"

void play_match(const std::string& model_path,
                const std::string& stockfish_path,
                int num_games,
                int time_per_move_ms,
                int engine_depth,
                int stockfish_skill,
                bool your_engine_plays_white,
                const std::string& eval_type)
{
    // … your full play_match body exactly as in the previous post …
    // (Initialize evaluator, Stockfish, loop over games, use board.gameOver(),
    //  alternate turns, detect checkmate/stalemate, tally results, etc.)
}
