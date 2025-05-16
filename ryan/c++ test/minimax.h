#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "evaluator.h"
#include <algorithm>

// Minimax search algorithm with alpha-beta pruning
inline float minimax(Board& board, int depth, float alpha, float beta, bool max, Evaluator& evaluator) {
    if (depth == 0 || board.gameOver()) {
        auto bbs = board.exportBitboards();
        return evaluator.eval(bbs) * (board.getTurn() ? 1.0f : -1.0f);
    }
    
    if (max) {
        float best = -1e9;
        auto moves = board.filterLegalMoves();
        
        // If no legal moves, return a very low score (similar to checkmate)
        if (moves.empty()) {
            return -1e8;
        }
        
        for (const auto& move : moves) {
            board.push(move);
            best = std::max(best, minimax(board, depth-1, alpha, beta, false, evaluator));
            board.pop();
            
            alpha = std::max(alpha, best);
            if (beta <= alpha) {
                break; // Beta cutoff
            }
        }
        return best;
    } else {
        float best = 1e9;
        auto moves = board.filterLegalMoves();
        
        // If no legal moves, return a very high score (similar to checkmate)
        if (moves.empty()) {
            return 1e8;
        }
        
        for (const auto& move : moves) {
            board.push(move);
            best = std::min(best, minimax(board, depth-1, alpha, beta, true, evaluator));
            board.pop();
            
            beta = std::min(beta, best);
            if (beta <= alpha) {
                break; // Alpha cutoff
            }
        }
        return best;
    }
}

#endif // MINIMAX_H
