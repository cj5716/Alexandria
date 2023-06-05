#pragma once
#include "board.h"
#include <map>
#include <string>
// convert ASCII character pieces to encoded constants
extern std::map<char, int> char_pieces;

// promoted pieces
extern std::map<int, char> promoted_pieces;

extern const int mvv_lva[12][12];

