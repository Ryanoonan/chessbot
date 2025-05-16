#pragma once

#include <string>
#include <sstream>
#include <array>
#include <cstdint>

// This file contains functions to convert between the internal bitboard representation
// and the FEN (Forsyth-Edwards Notation) string format, which is used for UCI communication.

inline std::string bitboards_to_fen(const std::array<uint64_t, 12>& bitboards, bool is_white_turn) {
    std::stringstream fen;
    char board[64] = {0};
    
    // Fill the board array with pieces
    for (int i = 0; i < 64; i++) {
        int square = i;
        
        for (int piece = 0; piece < 12; piece++) {
            if (bitboards[piece] & (1ULL << square)) {
                // 0-5 are white pieces (pawn, knight, bishop, rook, queen, king)
                // 6-11 are black pieces (pawn, knight, bishop, rook, queen, king)
                if (piece < 6) {
                    // White pieces - uppercase
                    const char white_pieces[6] = {'P', 'N', 'B', 'R', 'Q', 'K'};
                    board[square] = white_pieces[piece];
                } else {
                    // Black pieces - lowercase
                    const char black_pieces[6] = {'p', 'n', 'b', 'r', 'q', 'k'};
                    board[square] = black_pieces[piece - 6];
                }
                break;
            }
        }
    }
    
    // Build the FEN string for piece placement
    for (int rank = 0; rank < 8; rank++) {
        int empty_count = 0;
        
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;
            
            if (board[square] == 0) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    fen << empty_count;
                    empty_count = 0;
                }
                fen << board[square];
            }
        }
        
        if (empty_count > 0) {
            fen << empty_count;
        }
        
        if (rank < 7) {
            fen << '/';
        }
    }
    
    // Active color
    fen << (is_white_turn ? " w " : " b ");
    
    // Castling availability (simplified - we're not tracking this)
    fen << "- ";
    
    // En passant target square (simplified - we're not tracking this)
    fen << "- ";
    
    // Halfmove clock and fullmove number (simplified)
    fen << "0 1";
    
    return fen.str();
}

// Function to convert a chess.com notation move (e.g., "e2e4") to UCI notation
// This is just for clarity - in this case they're the same format
inline std::string chess_to_uci(const std::string& chess_move) {
    // In your case, the chess.com notation already matches UCI format
    // Remove any trailing promotion piece indicator
    if (chess_move.length() > 4) {
        std::string uci_move = chess_move.substr(0, 4);
        
        // Add promotion piece if present (lowercase)
        if (chess_move.length() > 4) {
            uci_move += chess_move.substr(4);
        }
        
        return uci_move;
    }
    
    return chess_move;
}
