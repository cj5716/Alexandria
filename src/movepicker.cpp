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

void InitMP(Movepicker *mp, S_Board* pos, Search_data* sd, Search_stack* ss, int ttMove, int threshold) {
    mp->pos = pos;
    mp->sd = sd;
    mp->ss = ss;
    mp->ttMove = MoveIsLegal(pos, ttMove) ? ttMove : NOMOVE;
    mp->threshold = threshold;
    mp->idx = 0;
    mp->stage = mp->ttMove ? PICK_TT : GEN_CAPTURES;
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

void ScoreQuiets(S_Board* pos, Search_data* sd, Search_stack* ss, S_MOVELIST* move_list) {
    // Loop through all the move in the movelist
    for (int i = 0; i < move_list->count; i++) {
        int move = move_list->moves[i].move;
        move_list->moves[i].score = GetHistoryScore(pos, sd, move, ss);
    }
}

int NextMove(Movepicker *mp, bool skipQuiets) {
top:
    if (mp->stage == PICK_TT) {
        ++mp->stage;
        mp->idx = 0;
        if (skipQuiets && IsQuiet(mp->ttMove))
            goto top;
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
        for (; mp->idx < mp->goodCaptures->count; ++mp->idx) {
            PickMove(mp->goodCaptures, mp->idx);
            int move = mp->goodCaptures->moves[mp->idx].move;

            if (move == mp->ttMove)
                continue;

            if (SEE(mp->pos, move, mp->threshold))
                return move;
            else
                AddMove(mp->badCaptures, move, mp->goodCaptures->moves[mp->idx].score);
        }
        ++mp->stage;
        mp->idx = 0;
        goto top;
    }
    else if (mp->stage == PICK_KILLER_0) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            mp->idx = 0;
            goto top;
        }
        ++mp->stage;
        int move = mp->ss->searchKillers[0];
        if (move != mp->ttMove && IsQuiet(move) && MoveIsLegal(mp->pos, move))
            return move;
        else {
            mp->idx = 0;
            goto top;
        }
    }
    else if (mp->stage == PICK_KILLER_1) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            goto top;
        }
        ++mp->stage;
        int move = mp->ss->searchKillers[1];
        if (move != mp->ttMove && IsQuiet(move) && MoveIsLegal(mp->pos, move))
            return move;
        else {
            mp->idx = 0;
            goto top;
        }
    }
    else if (mp->stage == PICK_COUNTER) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            goto top;
        }
        ++mp->stage;
        int countermove = mp->sd->CounterMoves[From((mp->ss - 1)->move)][To((mp->ss - 1)->move)];
        if (   countermove != mp->ttMove
            && countermove != mp->ss->searchKillers[0]
            && countermove != mp->ss->searchKillers[1]
            && IsQuiet(countermove)
            && MoveIsLegal(mp->pos, countermove))
            return countermove;
        else {
            mp->idx = 0;
            goto top;
        }
    }
    else if (mp->stage == GEN_QUIET) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            mp->idx = 0;
            goto top;
        }
        GenerateQuiets(mp->quiets, mp->pos);
        ScoreQuiets(mp->pos, mp->sd, mp->ss, mp->quiets);
        ++mp->stage;
        goto top;
    }
    else if (mp->stage == PICK_QUIET) {
        if (skipQuiets) {
            mp->stage = PICK_BAD_CAPTURES;
            mp->idx = 0;
            goto top;
        }
        for (; mp->idx < mp->quiets->count; ++mp->idx) {
            PickMove(mp->quiets, mp->idx);
            int move = mp->quiets->moves[mp->idx].move;
            if (   move == mp->ttMove 
                || move == mp->ss->searchKillers[0]
                || move == mp->ss->searchKillers[1]
                || move == mp->sd->CounterMoves[From((mp->ss - 1)->move)][To((mp->ss - 1)->move)])
                continue;

            return move;
        }
        ++mp->stage;
        goto top;
    }
    else if (mp->stage == PICK_BAD_CAPTURES) {
        for (; mp->idx < mp->badCaptures->count; ++mp->idx) {
            PickMove(mp->badCaptures, mp->idx);
            int move = mp->badCaptures->moves[mp->idx].move;
            if (move == mp->ttMove)
                continue;

            return move;
        }
        return NOMOVE;
    }
    return NOMOVE;
}
