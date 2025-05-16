// main.cpp
#include <iostream>
#include <string>
#include "engine_vs_stockfish.h"

int main(int argc, char* argv[]) {
    std::string model_path      = "build/chessnet_model.pt";
    std::string stockfish_path  = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish";
    int num_games       = 2;
    int time_per_move_ms= 1000;
    int engine_depth    = 3;
    int stockfish_skill = 5;
    bool your_engine_plays_white = true;
    std::string eval_type = "neural";

    // parse argv...
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // ... same as before ...
    }

    play_match(model_path,
               stockfish_path,
               num_games,
               time_per_move_ms,
               engine_depth,
               stockfish_skill,
               your_engine_plays_white,
               eval_type);
    return 0;
}
