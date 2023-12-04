#include "movegen.h"
#include "search.h"
#include "types.h"

#pragma once

struct S_MOVELIST;

enum {
    PICK_TT,
    GEN_CAPTURES,
    PICK_GOOD_CAPTURES,
    GEN_QUIET,
    PICK_QUIET,
    PICK_BAD_CAPTURES
};

struct Movepicker {
    S_Board* pos;
    Search_data* sd;
    Search_stack* ss;
    S_MOVELIST goodCaptures[1];
    S_MOVELIST quiets[1];
    S_MOVELIST badCaptures[1];
    int ttMove;
    int killer0;
    int killer1;
    int counter;
    int threshold;
    int idx;
    int stage;
    bool capturesOnly;
};

void InitMP(Movepicker *mp, S_Board* pos, Search_data* sd, Search_stack* ss, int ttMove, int killer0, int killer1, int counter, int threshold, bool capturesOnly);
int NextMove(Movepicker *mp, bool skipQuiets);