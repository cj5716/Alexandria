#include "movegen.h"
#include "search.h"
#include "types.h"

#pragma once

struct S_MOVELIST;

enum {
    PICK_TT,
    GEN_MOVES,
    PICK_MOVES
};

struct Movepicker {
    S_Board* pos;
    Search_data* sd;
    Search_stack* ss;
    S_MOVELIST moveList[1];
    int ttMove;
    int killer0;
    int killer1;
    int counter;
    int threshold;
    int idx;
    int stage;
    bool capturesOnly;
};

void InitMP(Movepicker *mp, S_Board* pos, Search_data* sd, Search_stack* ss, int ttMove, int threshold, bool capturesOnly);
int NextMove(Movepicker *mp, bool skipQuiets, bool skipNonGood);