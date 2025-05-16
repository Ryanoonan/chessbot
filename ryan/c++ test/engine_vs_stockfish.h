// engine_vs_stockfish.h
#ifndef ENGINE_VS_STOCKFISH_H
#define ENGINE_VS_STOCKFISH_H

#include <string>

// Declaration only
void play_match(const std::string& model_path,
                const std::string& stockfish_path,
                int num_games,
                int time_per_move_ms,
                int engine_depth,
                int stockfish_skill,
                bool your_engine_plays_white,
                const std::string& eval_type);

#endif // ENGINE_VS_STOCKFISH_H
