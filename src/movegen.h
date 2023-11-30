#pragma once

struct S_Board;
struct S_MOVELIST;

// is the square given in input attacked by the current given side
[[nodiscard]] bool IsSquareAttacked(const S_Board* pos, const int square, const int side);

// Check for move legality by generating the list of legal moves in a position and checking if that move is present
[[nodiscard]] bool MoveExists(const S_Board* pos, const int move);

// Generate all moves
void GenerateMoves(S_MOVELIST* move_list, S_Board* pos);

// Generate all captures
void GenerateCaptures(S_MOVELIST* move_list, S_Board* pos);

// Generate all quiets
void GenerateQuiets(S_MOVELIST* move_list, S_Board* pos);

// Check for move legality
bool MoveIsLegal(S_Board* pos, const int move);

// Add move to movelist
void AddMove(S_MOVELIST* list, const int move, const int score);
