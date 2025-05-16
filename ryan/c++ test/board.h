#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <vector>
#include <string>
#include "fen_utils.h"

class Board {
public:
    struct Move {
        int from, to, promotion_piece;
        bool is_capture;
        int captured_piece;
        Move(int f=0,int t=0,int promo=0,bool capt=false,int cap=-1);
        std::string to_string() const;
        static Move from_uci(const std::string& uci);
    };

    Board();

    std::array<uint64_t,12> exportBitboards() const;
    bool gameOver() const;
    bool getTurn() const;
    std::string getFen() const;
    bool isInCheck(bool whiteKing) const;

    std::vector<Move> legalMoves() const;
    std::vector<Move> filterLegalMoves() const;

    void push(const Move& m);
    void pop();

    std::vector<std::string> getMoveHistory() const;

private:
    std::array<uint64_t,12> bitboards;
    bool is_white_turn;
    std::vector<Move> move_history;

    bool isSquareOccupied(int sq) const;
    bool hasEnemyPiece(int sq, bool white) const;
};

#endif // BOARD_H
