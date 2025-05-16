#include "board.h"
#include <iostream>
#include <cmath>

// --- Move definitions ---
Board::Move::Move(int f,int t,int promo,bool capt,int cap)
  : from(f), to(t), promotion_piece(promo),
    is_capture(capt), captured_piece(cap) {}

std::string Board::Move::to_string() const {
    const char files[8] = {'a','b','c','d','e','f','g','h'};
    const char* pcs[5] = {"","n","b","r","q"};
    int ff=from%8, fr=from/8, tf=to%8, tr=to/8;
    std::string s;
    s += files[ff];
    s += std::to_string(8-fr);
    s += files[tf];
    s += std::to_string(8-tr);
    if (promotion_piece>0) s += pcs[promotion_piece];
    return s;
}

Board::Move Board::Move::from_uci(const std::string& uci){
    if (uci.size()<4) return Move();
    int ff=uci[0]-'a', fr=8-(uci[1]-'0');
    int tf=uci[2]-'a', tr=8-(uci[3]-'0');
    int f=fr*8+ff, t=tr*8+tf, promo=0;
    if (uci.size()>4){
      char p=uci[4];
      if(p=='n') promo=1;
      if(p=='b') promo=2;
      if(p=='r') promo=3;
      if(p=='q') promo=4;
    }
    return Move(f,t,promo);
}

// --- Board ctor & queries ---
Board::Board(){
    // White pieces
    bitboards[0]=0x00FF000000000000ULL;
    bitboards[1]=0x4200000000000000ULL;
    bitboards[2]=0x2400000000000000ULL;
    bitboards[3]=0x8100000000000000ULL;
    bitboards[4]=0x0800000000000000ULL;
    bitboards[5]=0x1000000000000000ULL;
    // Black pieces
    bitboards[6]=0x000000000000FF00ULL;
    bitboards[7]=0x0000000000000042ULL;
    bitboards[8]=0x0000000000000024ULL;
    bitboards[9]=0x0000000000000081ULL;
    bitboards[10]=0x0000000000000008ULL;
    bitboards[11]=0x0000000000000010ULL;
    is_white_turn=true;
}

std::array<uint64_t,12> Board::exportBitboards() const {
    return bitboards;
}

bool Board::getTurn() const {
    return is_white_turn;
}

std::string Board::getFen() const {
    return bitboards_to_fen(bitboards, is_white_turn);
}
bool Board::gameOver() const {
    // If a king is actually missing, that’s an immediate end
    if (bitboards[5]==0 || bitboards[11]==0) return true;

    // Otherwise if there are no legal moves, it’s checkmate or stalemate
    return filterLegalMoves().empty();
}

bool Board::isInCheck(bool whiteKing) const {
    uint64_t kbb=bitboards[whiteKing?5:11];
    if (!kbb) return false;
    int ksq=__builtin_ctzll(kbb);
    return hasEnemyPiece(ksq, !whiteKing);
}

// --- Move generation (pawns, knights, king) ---
bool Board::isSquareOccupied(int sq) const {
    for(auto bb:bitboards)
        if(bb & (1ULL<<sq)) return true;
    return false;
}

bool Board::hasEnemyPiece(int sq,bool white) const {
    int st=white?6:0, en=white?12:6;
    for(int i=st;i<en;i++)
        if(bitboards[i]&(1ULL<<sq)) return true;
    return false;
}

std::vector<Board::Move> Board::legalMoves() const {
    std::vector<Move> out;
    int off = is_white_turn ? 0 : 6;
    // Pawns
    uint64_t paw = bitboards[off];
    int dir = is_white_turn ? -8 : 8, sr = is_white_turn ? 6 : 1;
    while(paw) {
      int sq = __builtin_ctzll(paw);
      paw &= paw-1;
      int t1 = sq + dir;
      if (t1>=0 && t1<64 && !isSquareOccupied(t1)) {
        out.emplace_back(sq, t1);
        if (sq/8 == sr) {
          int t2 = t1 + dir;
          if (t2>=0 && t2<64 && !isSquareOccupied(t2))
            out.emplace_back(sq, t2);
        }
      }
      int f = sq % 8;
      if (f>0) {
        int lt = sq + dir - 1;
        if (lt>=0 && lt<64 && hasEnemyPiece(lt, is_white_turn))
          out.emplace_back(sq, lt, 0, true);
      }
      if (f<7) {
        int rt = sq + dir + 1;
        if (rt>=0 && rt<64 && hasEnemyPiece(rt, is_white_turn))
          out.emplace_back(sq, rt, 0, true);
      }
    }
    // Knights
    uint64_t kn = bitboards[off+1];
    const int km[8] = {-17,-15,-10,-6,6,10,15,17};
    while(kn) {
      int sq = __builtin_ctzll(kn);
      kn &= kn-1;
      int srk = sq/8, sf = sq%8;
      for(int d:km) {
        int tg = sq + d;
        if (tg<0||tg>63) continue;
        int tr=tg/8, tf=tg%8;
        if (std::abs(tr-srk)>2||std::abs(tf-sf)>2) continue;
        if (std::abs(tr-srk)+std::abs(tf-sf)!=3) continue;
        if (!isSquareOccupied(tg)) out.emplace_back(sq, tg);
        else if (hasEnemyPiece(tg, is_white_turn))
          out.emplace_back(sq, tg, 0, true);
      }
    }
    // King
    uint64_t kg = bitboards[off+5];
    const int k2[8] = {-9,-8,-7,-1,1,7,8,9};
    if (kg) {
      int sq = __builtin_ctzll(kg);
      int srk = sq/8, sf = sq%8;
      for(int d:k2) {
        int tg = sq + d;
        if (tg<0||tg>63) continue;
        int tr=tg/8, tf=tg%8;
        if (std::abs(tr-srk)>1||std::abs(tf-sf)>1) continue;
        if (!isSquareOccupied(tg)) out.emplace_back(sq, tg);
        else if (hasEnemyPiece(tg, is_white_turn))
          out.emplace_back(sq, tg, 0, true);
      }
    }
    return out;
}

std::vector<Board::Move> Board::filterLegalMoves() const {
    auto pseudo = legalMoves();
    std::vector<Move> legal;
    for(auto &m : pseudo) {
      Board tmp = *this;
      bool was = tmp.getTurn();
      tmp.push(m);
      if (!tmp.isInCheck(was)) legal.push_back(m);
    }
    return legal;
}

// --- push / pop / history ---
void Board::push(const Move& mv){
    uint64_t fb = 1ULL<<mv.from, tb = 1ULL<<mv.to;
    int mi=-1;
    for(int i=0; i<12; i++)
      if (bitboards[i]&fb) { mi=i; break; }
    if (mi<0) return;
    int cap=-1;
    for(int j=0; j<12; j++)
      if (bitboards[j]&tb) { cap=j; break; }
    if (cap>=0) bitboards[cap] &= ~tb;
    bitboards[mi] &= ~fb;
    bitboards[mi] |= tb;
    move_history.push_back({mv.from,mv.to,mv.promotion_piece,cap>=0,cap});
    is_white_turn = !is_white_turn;
}

void Board::pop(){
    if (move_history.empty()) return;
    auto mv = move_history.back();
    move_history.pop_back();
    uint64_t fb = 1ULL<<mv.from, tb = 1ULL<<mv.to;
    int mi=-1;
    for(int i=0; i<12; i++)
      if (bitboards[i]&tb) { mi=i; break; }
    if (mi>=0) {
      bitboards[mi] &= ~tb;
      bitboards[mi] |= fb;
    }
    if (mv.is_capture)
      bitboards[mv.captured_piece] |= tb;
    is_white_turn = !is_white_turn;
}

std::vector<std::string> Board::getMoveHistory() const {
    std::vector<std::string> out;
    for(auto &m: move_history)
      out.push_back(m.to_string());
    return out;
}
