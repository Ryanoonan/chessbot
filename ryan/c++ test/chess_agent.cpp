#include <torch/script.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <array>
#include <algorithm>
#include <random>
#include "chessnet_evaluator.h"
#include "uci_engine.h"
#include "fen_utils.h"

// Simple chess board structure
class Board {
private:
    std::array<uint64_t, 12> bitboards;
    bool is_white_turn;
    
public:
    struct Move {
        int from;
        int to;
        int promotion_piece; // 0 = none, 1-4 = knight, bishop, rook, queen
        bool is_capture;
        int captured_piece; // Index of captured piece (0-11), or -1 if none
        
        Move(int f = 0, int t = 0, int promo = 0, bool capt = false, int captPiece = -1)
            : from(f), to(t), promotion_piece(promo), is_capture(capt), captured_piece(captPiece) {}
        
        std::string to_string() const {
            const char files[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
            const char* pieces[5] = {"", "n", "b", "r", "q"};
            
            int from_file = from % 8;
            int from_rank = from / 8;
            int to_file = to % 8;
            int to_rank = to / 8;
            
            std::string result = "";
            result += files[from_file];
            result += std::to_string(8 - from_rank);  // Convert to chess notation (8-1)
            result += files[to_file];
            result += std::to_string(8 - to_rank);    // Convert to chess notation (8-1)
            
            if (promotion_piece > 0) {
                result += pieces[promotion_piece];
            }
            
            return result;
        }
        
        // Create a Move from UCI string (e.g., "e2e4", "a7a8q")
        static Move from_uci(const std::string& uci) {
            if (uci.size() < 4) return Move(); // Invalid move
            
            int from_file = uci[0] - 'a';
            int from_rank = 8 - (uci[1] - '0');
            int to_file = uci[2] - 'a';
            int to_rank = 8 - (uci[3] - '0');
            
            int from = from_rank * 8 + from_file;
            int to = to_rank * 8 + to_file;
            
            int promotion = 0;
            if (uci.size() > 4) {
                char promo = uci[4];
                switch(promo) {
                    case 'n': promotion = 1; break;
                    case 'b': promotion = 2; break;
                    case 'r': promotion = 3; break;
                    case 'q': promotion = 4; break;
                }
            }
            
            // Capture and captured piece will be determined by the board when the move is made
            return Move(from, to, promotion);
        }
    };
    std::vector<Move> move_history;

    Board() {
        // Initialize standard chess position
        // White pieces (pawns, knights, bishops, rooks, queens, kings)
        bitboards[0] = 0x00FF000000000000ULL;  // White pawns
        bitboards[1] = 0x4200000000000000ULL;  // White knights
        bitboards[2] = 0x2400000000000000ULL;  // White bishops
        bitboards[3] = 0x8100000000000000ULL;  // White rooks
        bitboards[4] = 0x0800000000000000ULL;  // White queen
        bitboards[5] = 0x1000000000000000ULL;  // White king
        
        // Black pieces (pawns, knights, bishops, rooks, queens, kings)
        bitboards[6] = 0x000000000000FF00ULL;  // Black pawns
        bitboards[7] = 0x0000000000000042ULL;  // Black knights
        bitboards[8] = 0x0000000000000024ULL;  // Black bishops
        bitboards[9] = 0x0000000000000081ULL;  // Black rooks
        bitboards[10] = 0x0000000000000008ULL; // Black queen
        bitboards[11] = 0x0000000000000010ULL; // Black king
        
        is_white_turn = true;
    }
    
    std::array<uint64_t, 12> exportBitboards() const {
        return bitboards;
    }
    
    bool gameOver() const {
        // This is a simplified check - in a real implementation,
        // you'd check for checkmate, stalemate, etc.
        bool white_king_present = bitboards[5] != 0;
        bool black_king_present = bitboards[11] != 0;
        return !white_king_present || !black_king_present;
    }
    
    bool getTurn() const {
        return is_white_turn;
    }
    
    std::string getFen() const {
        return bitboards_to_fen(bitboards, is_white_turn);
    }
    
    std::vector<std::string> getMoveHistory() const {
        std::vector<std::string> history;
        for (const auto& move : move_history) {
            history.push_back(move.to_string());
        }
        return history;
    }
    
    // Helper method to check if a square is occupied
    bool isSquareOccupied(int square) const {
        for (int i = 0; i < 12; i++) {
            if (bitboards[i] & (1ULL << square)) {
                return true;
            }
        }
        return false;
    }
    
    // Helper method to check if a square has an enemy piece
    bool hasEnemyPiece(int square, bool is_white) const {
        int start_idx = is_white ? 6 : 0;
        int end_idx = is_white ? 12 : 6;
        
        for (int i = start_idx; i < end_idx; i++) {
            if (bitboards[i] & (1ULL << square)) {
                return true;
            }
        }
        return false;
    }
    
    // Generate pseudo-legal moves
    std::vector<Move> legalMoves() const {
        std::vector<Move> moves;
        
        // Determine whose turn it is
        int piece_offset = is_white_turn ? 0 : 6;
        
        // Generate pawn moves
        uint64_t pawns = bitboards[piece_offset]; // Get pawns
        int direction = is_white_turn ? -8 : 8;   // Direction pawns move
        int start_rank = is_white_turn ? 6 : 1;   // Starting rank for pawns
        
        while (pawns) {
            int sq = __builtin_ctzll(pawns); // Get least significant bit position
            pawns &= pawns - 1;              // Clear least significant bit
            
            int rank = sq / 8;
            int file = sq % 8;
            
            // One square forward
            int target = sq + direction;
            if (target >= 0 && target < 64 && !isSquareOccupied(target)) {
                moves.push_back(Move(sq, target, 0, false));
                
                // Two squares forward from starting position
                if (rank == start_rank) {
                    int double_target = target + direction;
                    if (double_target >= 0 && double_target < 64 && !isSquareOccupied(double_target)) {
                        moves.push_back(Move(sq, double_target, 0, false));
                    }
                }
            }
            
            // Captures diagonally
            if (file > 0) { // Can capture to the left
                int left_capture = sq + direction - 1;
                if (left_capture >= 0 && left_capture < 64 && 
                    hasEnemyPiece(left_capture, is_white_turn)) {
                    moves.push_back(Move(sq, left_capture, 0, true));
                }
            }
            
            if (file < 7) { // Can capture to the right
                int right_capture = sq + direction + 1;
                if (right_capture >= 0 && right_capture < 64 && 
                    hasEnemyPiece(right_capture, is_white_turn)) {
                    moves.push_back(Move(sq, right_capture, 0, true));
                }
            }
        }
        
        // Generate knight moves
        uint64_t knights = bitboards[piece_offset + 1]; // Get knights
        const int knight_moves[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
        
        while (knights) {
            int sq = __builtin_ctzll(knights);
            knights &= knights - 1;
            
            for (int i = 0; i < 8; i++) {
                int target = sq + knight_moves[i];
                if (target >= 0 && target < 64) {
                    // Verify it's a valid knight move (max 2 squares in any direction)
                    int src_rank = sq / 8;
                    int src_file = sq % 8;
                    int dst_rank = target / 8;
                    int dst_file = target % 8;
                    
                    if (abs(src_rank - dst_rank) > 2 || abs(src_file - dst_file) > 2) {
                        continue;
                    }
                    
                    if (abs(src_rank - dst_rank) + abs(src_file - dst_file) != 3) {
                        continue;
                    }
                    
                    // Check if the target square is empty or has an enemy piece
                    if (!isSquareOccupied(target)) {
                        moves.push_back(Move(sq, target, 0, false));
                    } else if (hasEnemyPiece(target, is_white_turn)) {
                        moves.push_back(Move(sq, target, 0, true));
                    }
                }
            }
        }
        
        // Generate king moves (simplified, no castling)
        uint64_t king = bitboards[piece_offset + 5]; // Get king
        const int king_moves[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
        
        if (king) {
            int sq = __builtin_ctzll(king);
            
            for (int i = 0; i < 8; i++) {
                int target = sq + king_moves[i];
                if (target >= 0 && target < 64) {
                    // Verify it's a valid king move (max 1 square in any direction)
                    int src_rank = sq / 8;
                    int src_file = sq % 8;
                    int dst_rank = target / 8;
                    int dst_file = target % 8;
                    
                    if (abs(src_rank - dst_rank) > 1 || abs(src_file - dst_file) > 1) {
                        continue;
                    }
                    
                    // Check if the target square is empty or has an enemy piece
                    if (!isSquareOccupied(target)) {
                        moves.push_back(Move(sq, target, 0, false));
                    } else if (hasEnemyPiece(target, is_white_turn)) {
                        moves.push_back(Move(sq, target, 0, true));
                    }
                }
            }
        }
        
        // In a complete implementation, you would add move generation for other pieces
        
        return moves;
    }
    
    void push(const Move& move) {
        // Create a copy of the move with additional info for proper move undo
        Move stored_move = move;
        
        uint64_t from_bit = 1ULL << move.from;
        uint64_t to_bit = 1ULL << move.to;
        
        // Find which piece is moving
        int moving_piece_index = -1;
        for (int i = 0; i < 12; i++) {
            if (bitboards[i] & from_bit) {
                moving_piece_index = i;
                break;
            }
        }
        
        if (moving_piece_index < 0) {
            std::cerr << "Error: No piece found at square " << move.from << std::endl;
            return;
        }
        
        // Check for capture and store captured piece info
        stored_move.captured_piece = -1;
        for (int j = 0; j < 12; j++) {
            if (bitboards[j] & to_bit) {
                stored_move.captured_piece = j;
                stored_move.is_capture = true;
                break;
            }
        }
        
        // Remove piece from 'from' square
        bitboards[moving_piece_index] &= ~from_bit;
        
        // Remove captured piece if any
        if (stored_move.captured_piece >= 0) {
            bitboards[stored_move.captured_piece] &= ~to_bit;
        }
        
        // Place piece on 'to' square
        bitboards[moving_piece_index] |= to_bit;
        
        // Store move in history (with capture info)
        move_history.push_back(stored_move);
        
        // Switch turns
        is_white_turn = !is_white_turn;
    }
    
    void pop() {
        // Undo the last move
        if (!move_history.empty()) {
            Move last_move = move_history.back();
            move_history.pop_back();
            
            uint64_t from_bit = 1ULL << last_move.from;
            uint64_t to_bit = 1ULL << last_move.to;
            
            // Find which piece is on the 'to' square
            int piece_index = -1;
            for (int i = 0; i < 12; i++) {
                if (bitboards[i] & to_bit) {
                    piece_index = i;
                    break;
                }
            }
            
            if (piece_index >= 0) {
                // Move piece back to 'from' square
                bitboards[piece_index] &= ~to_bit;
                bitboards[piece_index] |= from_bit;
                
                // If there was a capture, restore the captured piece
                if (last_move.is_capture && last_move.captured_piece >= 0) {
                    bitboards[last_move.captured_piece] |= to_bit;
                }
            } else {
                std::cerr << "Error: No piece found at square " << last_move.to << " during undo" << std::endl;
            }
            
            // Switch turns back
            is_white_turn = !is_white_turn;
        }
    }
    
    void print() const {
        std::cout << "  -----------------\n";
        for (int rank = 0; rank < 8; rank++) {
            std::cout << (8 - rank) << " |"; // Print rank number (8-1)
            for (int file = 0; file < 8; file++) {
                int square = rank * 8 + file;
                char piece = '.';
                
                // Find which piece is on this square
                for (int i = 0; i < 12; i++) {
                    if (bitboards[i] & (1ULL << square)) {
                        // Map index to piece character
                        const char pieces[12] = {'P', 'N', 'B', 'R', 'Q', 'K', 
                                                 'p', 'n', 'b', 'r', 'q', 'k'};
                        piece = pieces[i];
                        break;
                    }
                }
                
                std::cout << piece << "|";
            }
            std::cout << "\n  -----------------\n";
        }
        std::cout << "   a b c d e f g h\n";
        std::cout << "Turn: " << (is_white_turn ? "White" : "Black") << "\n\n";
    }
};

// Minimax search with alpha-beta pruning
float minimax(Board& board, int depth, float alpha, float beta, bool max, ChessNetEvaluator& evaluator) {
    if (depth == 0 || board.gameOver()) {
        auto bbs = board.exportBitboards();
        return evaluator.eval(bbs) * (board.getTurn() ? 1.0f : -1.0f);
    }
    
    if (max) {
        float best = -1e9;
        auto moves = board.legalMoves();
        
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
        auto moves = board.legalMoves();
        
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

// Function to find the best move
Board::Move find_best_move(Board& board, int depth, ChessNetEvaluator& evaluator) {
    auto moves = board.legalMoves();
    
    // If no legal moves, return an invalid move
    if (moves.empty()) {
        return Board::Move(-1, -1, 0, false);
    }
    
    float best_score = -1e9;
    Board::Move best_move = moves[0];
    bool is_maximizing = board.getTurn(); // White is maximizing
    
    // For randomizing equally good moves
    std::vector<Board::Move> best_moves;
    
    for (const auto& move : moves) {
        board.push(move);
        float score = minimax(board, depth-1, -1e9, 1e9, !is_maximizing, evaluator);
        board.pop();
        
        if (score > best_score) {
            best_score = score;
            best_moves.clear();
            best_moves.push_back(move);
        } else if (score == best_score) {
            best_moves.push_back(move);
        }
    }
    
    // Choose a random move from the best moves
    if (!best_moves.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, best_moves.size() - 1);
        best_move = best_moves[distrib(gen)];
    }
    
    std::cout << "Best move score: " << best_score << std::endl;
    return best_move;
}

// Function to find the best move using minimax (simplified minimax implementation for engine vs stockfish)
float minimax_stock(Board& board, int depth, float alpha, float beta, bool max, ChessNetEvaluator& evaluator) {
    if (depth == 0 || board.gameOver()) {
        auto bbs = board.exportBitboards();
        return evaluator.eval(bbs) * (board.getTurn() ? 1.0f : -1.0f);
    }
    
    if (max) {
        float best = -1e9;
        auto moves = board.legalMoves();
        
        // If no legal moves, return a very low score (similar to checkmate)
        if (moves.empty()) {
            return -1e8;
        }
        
        for (const auto& move : moves) {
            board.push(move);
            best = std::max(best, minimax_stock(board, depth-1, alpha, beta, false, evaluator));
            board.pop();
            
            alpha = std::max(alpha, best);
            if (beta <= alpha) {
                break; // Beta cutoff
            }
        }
        return best;
    } else {
        float best = 1e9;
        auto moves = board.legalMoves();
        
        // If no legal moves, return a very high score (similar to checkmate)
        if (moves.empty()) {
            return 1e8;
        }
        
        for (const auto& move : moves) {
            board.push(move);
            best = std::min(best, minimax_stock(board, depth-1, alpha, beta, true, evaluator));
            board.pop();
            
            beta = std::min(beta, best);
            if (beta <= alpha) {
                break; // Alpha cutoff
            }
        }
        return best;
    }
}

// Play a match between your chess engine and Stockfish
void play_match(const std::string& model_path, const std::string& stockfish_path, 
                int num_games = 2, int time_per_move_ms = 10, int engine_depth = 3,
                int stockfish_skill = 5, bool your_engine_plays_white = true) {
    
    // Results tracking
    int your_engine_wins = 0;
    int stockfish_wins = 0;
    int draws = 0;
    
    // Initialize your engine's evaluation function
    std::cout << "Loading neural network model from " << model_path << "...\n";
    ChessNetEvaluator evaluator(model_path);
    
    // Initialize Stockfish
    std::cout << "Initializing Stockfish from " << stockfish_path << "...\n";
    UCI_Engine stockfish(stockfish_path, stockfish_skill);
    if (!stockfish.initialize()) {
        std::cerr << "Failed to initialize Stockfish engine. Exiting.\n";
        return;
    }
    
    // Set Stockfish skill level
    stockfish.write_command("setoption name Skill Level value " + std::to_string(stockfish_skill));
    
    for (int game = 1; game <= num_games; game++) {
        std::cout << "\n======= Game " << game << " of " << num_games << " =======\n";
        
        // Initialize a new board
        Board board;
        board.print();
        
        // Determine who plays which color
        bool engine_is_white = (game % 2 == 1) ? your_engine_plays_white : !your_engine_plays_white;
        std::cout << "Your Engine plays: " << (engine_is_white ? "White" : "Black") << "\n";
        std::cout << "Stockfish plays: " << (engine_is_white ? "Black" : "White") << "\n\n";
        
        // Game loop
        int move_count = 0;
        while (!board.gameOver()) {
            move_count++;
            
            bool your_engine_turn = (board.getTurn() == engine_is_white);
            std::string fen = board.getFen();
            std::vector<std::string> move_history = board.getMoveHistory();
            
            std::cout << "Move " << move_count << ": ";
            
            if (your_engine_turn) {
                // Your engine's turn
                std::cout << "Your Engine is thinking...\n";
                
                // Get best move using minimax search
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // Get legal moves from the current position and choose the best one
                std::vector<Board::Move> legal_moves = board.legalMoves();
                
                if (legal_moves.empty()) {
                    std::cout << "No legal moves available. Game over.\n";
                    break;
                }
                
                float best_score = -1e9;
                Board::Move best_move = legal_moves[0];
                
                for (const auto& move : legal_moves) {
                    board.push(move);
                    float score = minimax_stock(board, engine_depth - 1, -1e9, 1e9, !board.getTurn(), evaluator);
                    board.pop();
                    
                    if (score > best_score) {
                        best_score = score;
                        best_move = move;
                    }
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                // Apply the move
                std::string move_str = best_move.to_string();
                std::cout << "Your Engine played: " << move_str << " (in " << duration.count() << "ms)\n";
                board.push(best_move);
            }
            else {
                // Stockfish's turn
                std::cout << "Stockfish is thinking...\n";
                
                // Set position in Stockfish
                stockfish.set_position(fen, move_history);
                
                // Get Stockfish's move
                std::string sf_move = stockfish.get_best_move(time_per_move_ms);
                std::cout << "Stockfish played: " << sf_move << "\n";
                
                // Apply Stockfish's move to our board
                Board::Move move = Board::Move::from_uci(sf_move);
                board.push(move);
            }
            
            // Print the updated board
            board.print();
            
            // Short delay for readability
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        // Game result
        std::cout << "\nGame " << game << " completed after " << move_count << " moves.\n";
        
        // Determine winner (simplified)
        bool white_king_present = (board.exportBitboards()[5] != 0);
        bool black_king_present = (board.exportBitboards()[11] != 0);
        
        if (!white_king_present && !black_king_present) {
            std::cout << "Game ended in a draw (both kings captured).\n";
            draws++;
        }
        else if (!white_king_present) {
            std::cout << "Black wins!\n";
            if (engine_is_white) stockfish_wins++;
            else your_engine_wins++;
        }
        else if (!black_king_present) {
            std::cout << "White wins!\n";
            if (engine_is_white) your_engine_wins++;
            else stockfish_wins++;
        }
        else {
            std::cout << "Game ended in a draw (no kings captured).\n";
            draws++;
        }
    }
    
    // Print results
    std::cout << "\n======= Match Results =======\n";
    std::cout << "Your Engine: " << your_engine_wins << " wins\n";
    std::cout << "Stockfish: " << stockfish_wins << " wins\n";
    std::cout << "Draws: " << draws << " games\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "chessnet_model.pt";
    int depth = 3;
    std::string mode = "self_play"; // Default mode
    std::string stockfish_path = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish";
    int num_games = 2;
    int time_per_move_ms = 10;
    int stockfish_skill = 5;
    bool your_engine_plays_white = true;
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        } else if (arg == "--depth" && i + 1 < argc) {
            depth = std::stoi(argv[i + 1]);
            i++;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[i + 1];
            i++;
        } else if (arg == "--stockfish" && i + 1 < argc) {
            stockfish_path = argv[i + 1];
            i++;
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[i + 1]);
            i++;
        } else if (arg == "--time" && i + 1 < argc) {
            time_per_move_ms = std::stoi(argv[i + 1]);
            i++;
        } else if (arg == "--skill" && i + 1 < argc) {
            stockfish_skill = std::stoi(argv[i + 1]);
            i++;
        } else if (arg == "--color" && i + 1 < argc) {
            std::string color = argv[i + 1];
            your_engine_plays_white = (color == "white");
            i++;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --model PATH       Path to your neural network model (default: chessnet_model.pt)\n";
            std::cout << "  --depth N          Search depth for your engine (default: 3)\n";
            std::cout << "  --mode MODE        Game mode: 'self_play' or 'engine_vs_stockfish' (default: self_play)\n";
            std::cout << "  --stockfish PATH   Path to Stockfish executable (default: /opt/homebrew/Cellar/stockfish/17/bin/stockfish)\n";
            std::cout << "  --games N          Number of games to play (default: 2)\n";
            std::cout << "  --time MS          Time per move in milliseconds (default: 1000)\n";
            std::cout << "  --skill N          Stockfish skill level (0-20, default: 5)\n";
            std::cout << "  --color COLOR      Your engine's color in first game ('white' or 'black', default: white)\n";
            std::cout << "  --help             Display this help message\n";
            return 0;
        }
    }
    
    std::cout << "Using model: " << model_path << std::endl;
    
    if (mode == "engine_vs_stockfish") {
        try {
            // Play match between your engine and Stockfish
            play_match(model_path, stockfish_path, num_games, time_per_move_ms, depth, stockfish_skill, your_engine_plays_white);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        // Default mode: self-play
        std::cout << "Search depth: " << depth << std::endl;
    try {
        ChessNetEvaluator evaluator(model_path);
        
        // Create a new board with the starting position
        Board board;
        
        // Keep track of moves for chess.com format
        std::vector<std::string> move_sequence;
        int move_number = 1;
        
        // Main game loop
        while (!board.gameOver()) {
            // Print the current board
            board.print();
            
            if (board.getTurn()) {
                // White's turn (AI)
                std::cout << "AI is thinking..." << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                
                Board::Move best_move = find_best_move(board, depth, evaluator);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                std::string move_str = best_move.to_string();
                std::cout << "AI chose move: " << move_str 
                          << " (took " << duration.count() << "ms)" << std::endl;
                
                move_sequence.push_back(move_str);
                board.push(best_move);
            } else {
                // Black's turn (also AI for now, could be human)
                std::cout << "AI is thinking..." << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                
                Board::Move best_move = find_best_move(board, depth, evaluator);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                std::string move_str = best_move.to_string();
                std::cout << "AI chose move: " << move_str 
                          << " (took " << duration.count() << "ms)" << std::endl;
                
                move_sequence.push_back(move_str);
                board.push(best_move);
                
                // Increment move number after black's move
                move_number++;
            }
            
            // Add a small delay instead of waiting for Enter
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        board.print();
        std::cout << "Game over!" << std::endl;
        
        // Print all moves in chess.com format
        std::cout << "\n--- Game moves (copy-paste into chess.com) ---\n";
        
        // Print in standard algebraic notation format
        for (size_t i = 0; i < move_sequence.size(); i++) {
            if (i % 2 == 0) {
                std::cout << (i / 2) + 1 << ". " << move_sequence[i];
            } else {
                std::cout << " " << move_sequence[i] << " ";
            }
        }
        std::cout << "\n\n";
        
        // Print all moves in a simple space-separated format
        std::cout << "--- Moves (simple format) ---\n";
        for (const auto& move : move_sequence) {
            std::cout << move << " ";
        }
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    }
    
    return 0;
}
