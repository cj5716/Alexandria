#include "history.h"
#include "movegen.h"
#include "movepicker.h"
#include "move.h"
#include "piece_data.h"
#include "search.h"
#include "types.h"

void PickMove(S_MOVELIST* moveList, const int moveNum) {
    int bestScore = -2147483645;
    int bestNum = moveNum;
    // starting at the number of the current move and stopping at the end of the list
    for (int index = moveNum; index < moveList->count; ++index) {
        // if we find a move with a better score than our bestmove we use that as the new best move
        if (moveList->moves[index].score > bestScore) {
            bestScore = moveList->moves[index].score;
            bestNum = index;
        }
    }
    // swap the move with the best score with the move in place moveNum
    S_MOVE temp = moveList->moves[moveNum];
    moveList->moves[moveNum] = moveList->moves[bestNum];
    moveList->moves[bestNum] = temp;
}

void InitMP(Movepicker* mp, S_Board* pos, Search_data* sd, Search_stack* ss, int ttMove, int killer0, int killer1, int counter, int threshold, bool capturesOnly) {
    mp->pos = pos;
    mp->sd = sd;
    mp->ss = ss;
    mp->ttMove = (!capturesOnly || !IsQuiet(ttMove)) && MoveIsLegal(pos, ttMove) ? ttMove : NOMOVE;
    mp->killer0 = killer0;
    mp->killer1 = killer1;
    mp->counter = counter;
    mp->threshold = threshold;
    mp->idx = 0;
    mp->stage = mp->ttMove ? PICK_TT : GEN_CAPTURES;
    mp->capturesOnly = capturesOnly;
    std::memset(mp->goodCaptures, 0, sizeof(mp->goodCaptures));
    std::memset(mp->quiets, 0, sizeof(mp->quiets));
    std::memset(mp->badCaptures, 0, sizeof(mp->badCaptures));
}

void ScoreCaptures(S_Board* pos, Search_data* sd, S_MOVELIST* move_list) {
    // Loop through all the move in the movelist
    for (int i = 0; i < move_list->count; i++) {
        int move = move_list->moves[i].move;
        if (isPromo(move)) {
            switch (getPromotedPiecetype(move)) {
            case QUEEN:
                move_list->moves[i].score = queenPromotionScore;
                break;
            case KNIGHT:
                move_list->moves[i].score = knightPromotionScore;
                break;
            case ROOK:
                move_list->moves[i].score = badPromotionScore;
                break;
            case BISHOP:
                move_list->moves[i].score = badPromotionScore;
                break;
            default:
                break;
            }
        }
        // Capture
        else {
            int captured_piece = isEnpassant(move) ? PAWN : GetPieceType(pos->PieceOn(To(move)));
            // Sort by most valuable victim and capthist, with LVA as tiebreaks
            move_list->moves[i].score = mvv_lva[GetPieceType(Piece(move))][captured_piece] + GetCapthistScore(pos, sd, move);
            continue;
        }
    }
}

void ScoreQuiets(S_Board* pos, Search_data* sd, Search_stack* ss, S_MOVELIST* moveList, int killer0, int killer1, int counter) {
    // Loop through all the move in the movelist
    for (int i = 0; i < moveList->count; i++) {
        int move = moveList->moves[i].move;
        if (move == killer0)
            moveList->moves[i].score = killerMoveScore0;
        else if (move == killer1)
            moveList->moves[i].score = killerMoveScore1;
        else if (move == counter)
            moveList->moves[i].score = counterMoveScore;
        else
            moveList->moves[i].score = GetHistoryScore(pos, sd, move, ss);
    }
}

int NextMove(Movepicker* mp, bool skipQuiets) {
    skipQuiets |= mp->capturesOnly;
top:
    if (mp->stage == PICK_TT) {
        ++mp->stage;
        mp->idx = 0;
        return mp->ttMove;
    }
    else if (mp->stage == GEN_CAPTURES) {
        GenerateCaptures(mp->goodCaptures, mp->pos);
        ScoreCaptures(mp->pos, mp->sd, mp->goodCaptures);
        ++mp->stage;
        mp->idx = 0;
        goto top;
    }
    else if (mp->stage == PICK_GOOD_CAPTURES) {
        while (mp->idx < mp->goodCaptures->count) {
            PickMove(mp->goodCaptures, mp->idx);
            int move = mp->goodCaptures->moves[mp->idx].move;
            ++mp->idx;

            if (move == mp->ttMove)
                continue;

            if (SEE(mp->pos, move, mp->threshold) && mp->goodCaptures->moves[mp->idx].score != badPromotionScore)
                return move;
            else
                AddMove(mp->badCaptures, move, mp->goodCaptures->moves[mp->idx].score);
        }
        ++mp->stage;
        mp->idx = 0;
        goto top;
    }
    else if (mp->stage == GEN_QUIET) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            mp->idx = 0;
            goto top;
        }
        GenerateQuiets(mp->quiets, mp->pos);
        ScoreQuiets(mp->pos, mp->sd, mp->ss, mp->quiets, mp->killer0, mp->killer1, mp->counter);
        ++mp->stage;
        goto top;
    }
    else if (mp->stage == PICK_QUIET) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            mp->idx = 0;
            goto top;
        }
        while (mp->idx < mp->quiets->count) {
            PickMove(mp->quiets, mp->idx);
            int move = mp->quiets->moves[mp->idx].move;
            ++mp->idx;
            if (move != mp->ttMove)
                return move;
        }
        ++mp->stage;
        mp->idx = 0;
        goto top;
    }
    else if (mp->stage == PICK_BAD_CAPTURES) {
        while (mp->idx < mp->badCaptures->count) {
            PickMove(mp->badCaptures, mp->idx);
            int move = mp->badCaptures->moves[mp->idx].move;
            ++mp->idx;
            if (move != mp->ttMove)
                return move;
        }
    }
    return NOMOVE;
}
