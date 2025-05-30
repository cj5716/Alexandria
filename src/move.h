#pragma once

#include "types.h"

struct Position;

struct ScoredMove {
    Move move;
    int score;
};

// move list structure
struct MoveList {
    ScoredMove moves[256];
    int count = 0;
};

// move list structure
struct StackMoveList {
    Move moves[64];
    int count = 0;
    void addMove(Move move){
        if(count < 64) {
            moves[count] = move;
            count++;
        }
    }
};

enum class Movetype {
    Quiet, doublePush, KSCastle,
    QSCastle, Capture, enPassant,
    knightPromo=8, bishopPromo, rookPromo, queenPromo,
    knightCapturePromo, bishopCapturePromo, rookCapturePromo, queenCapturePromo
};

inline Movetype operator | (Movetype first, Movetype second){
    return static_cast<Movetype>((static_cast<int>(first) | static_cast<int>(second)));
}

inline Move encode_move(const int source, const int target, const int piece, const Movetype movetype) {
    return (source) | (target << 6) | (static_cast<int>(movetype) << 12) | (piece << 16);
}

inline unsigned int From(const Move move) { return move & 0x3F; }
inline unsigned int To(const Move move) { return ((move & 0xFC0) >> 6); }
inline unsigned int FromTo(const Move move) { return move & 0xFFF; }
inline unsigned int Piece(const Move move) { return ((move & 0xF0000) >> 16); }
inline unsigned int PieceTo(const Move move) { return (Piece(move) << 6) | To(move); }
inline unsigned int PieceTypeTo(const Move move) { return (PieceType[Piece(move)] << 6) | To(move); }
inline unsigned int GetMovetype(const Move move) { return ((move & 0xF000) >> 12); }
inline unsigned int getPromotedPiecetype(const Move move) { return (GetMovetype(move) & 3) + 1; }
inline bool isEnpassant(const Move move) { return GetMovetype(move) == static_cast<int>(Movetype::enPassant); }
inline bool isDP(const Move move) { return GetMovetype(move) == static_cast<int>(Movetype::doublePush); }
inline bool isCastle(const Move move) { 
    return (GetMovetype(move) == static_cast<int>(Movetype::KSCastle)) || (GetMovetype(move) == static_cast<int>(Movetype::QSCastle));
}
inline bool isCapture(const Move move) { return GetMovetype(move) & static_cast<int>(Movetype::Capture); }
inline bool isQuiet(const Move move) { return !isCapture(move); }
inline bool isPromo(const Move move) { return GetMovetype(move) & 8; }
// Shorthand for captures + any promotion no matter if quiet or not 
inline bool isTactical(const Move move) { return isCapture(move) || isPromo(move); }
