#include "eval.h"
#include "board.h"
#include <algorithm>

// if we don't have enough material to mate consider the position a draw
bool MaterialDraw(const S_Board* pos) {
    // If we only have kings on the board then it's a draw
    if (pos->PieceCount() == 2)
        return true;
    // KN v K, KB v K
    else if (pos->PieceCount() == 3 && ((CountBits(GetPieceBB(pos, KNIGHT)) == 1) || (CountBits(GetPieceBB(pos, BISHOP)) == 1)))
        return true;
    // If we have 4 pieces on the board
    else if (pos->PieceCount() == 4) {
        // KNN v K, KN v KN
        if ((CountBits(GetPieceBB(pos, KNIGHT)) == 2))
            return true;
        // KB v KB
        else if (((CountBits(GetPieceBB(pos, BISHOP)) == 2)) && CountBits(pos->GetPieceColorBB(BISHOP, WHITE)) == 1)
            return true;
    }

    return false;
}

// position evaluation
int EvalPosition(S_Board* pos) {
    nnue.update(pos->AccumulatorTop(), pos->NNUEAdd, pos->NNUESub);
    bool stm = (pos->side == WHITE);
    int eval = nnue.output(pos->accumStack[pos->accumStackHead - 1], stm);
    int phase = GetGamePhase(pos);
    eval = (phase * eval * 118 + (64 - phase) * eval * 99) / 8192;
    eval = eval * (200 - pos->Get50mrCounter()) / 200;
    // Clamp eval to avoid it somehow being a mate score
    eval = std::clamp(eval, -mate_score + 1, mate_score - 1);
    return eval;
}
