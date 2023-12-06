#include "attack.h"
#include "board.h"
#include "init.h"
#include "magic.h"
#include "makemove.h"
#include "move.h"
#include "movegen.h"

// is the square given in input attacked by the current given side
bool IsSquareAttacked(const S_Board* pos, const int square, const int side) {
    // Take the occupancies of both positions, encoding where all the pieces on the board reside
    Bitboard occ = pos->Occupancy(BOTH);
    // is the square attacked by pawns
    if (pawn_attacks[side ^ 1][square] & pos->GetPieceColorBB(PAWN, side))
        return true;
    // is the square attacked by knights
    if (knight_attacks[square] & pos->GetPieceColorBB(KNIGHT, side))
        return true;
    // is the square attacked by kings
    if (king_attacks[square] & pos->GetPieceColorBB(KING, side))
        return true;
    // is the square attacked by bishops
    if (GetBishopAttacks(square, occ) & (pos->GetPieceColorBB(BISHOP, side) | pos->GetPieceColorBB(QUEEN, side)))
        return true;
    // is the square attacked by rooks
    if (GetRookAttacks(square, occ) & (pos->GetPieceColorBB(ROOK, side) | pos->GetPieceColorBB(QUEEN, side)))
        return true;
    // by default return false
    return false;
}

static inline Bitboard PawnPush(int color, int sq) {
    if (color == WHITE)
        return (1ULL << (sq - 8));
    return (1ULL << (sq + 8));
}

void init(S_Board* pos, int color, int sq) {
    Bitboard newMask = DoCheckmask(pos, color, sq);
    pos->checkMask = newMask ? newMask : 18446744073709551615ULL;
    DoPinMask(pos, color, sq);
}
// Check for move legality by generating the list of legal moves in a position and checking if that move is present
bool MoveExists(S_Board* pos, const int move) {
    S_MOVELIST list[1];
    GenerateMoves(list, pos);

    for (int moveNum = 0; moveNum < list->count; ++moveNum) {
        if (list->moves[moveNum].move == move) {
            return true;
        }
    }
    return false;
}

void AddMove(S_MOVELIST* list, const int move, const int score) {
    list->moves[list->count].move = move;
    list->moves[list->count].score = score;
    list->count++;
}

// function that adds a move to the move list
void AddMove(int move, S_MOVELIST* list) {
    AddMove(list, move, 0);
}
// function that adds a pawn move (and all its possible branches) to the move list
static inline void AddPawnMove(const S_Board* pos, const int from, const int to, S_MOVELIST* list) {
    Movetype movetype = pos->PieceOn(to) != EMPTY ? Movetype::Capture : Movetype::Quiet;
    if (!(abs(to - from) - 16)) movetype = Movetype::doublePush;
    else if (!(to - pos->enPas)) movetype = Movetype::enPassant;
    int pc = GetPiece(PAWN, pos->side);

    if ((1ULL << to) & 0xFF000000000000FFULL) { // if the pawn is moving from the 7th to the 8th rank
        AddMove(encode_move(from, to, pc, (Movetype::queenPromo | movetype)), list);
        AddMove(encode_move(from, to, pc, (Movetype::rookPromo | movetype)), list); // consider every possible piece promotion
        AddMove(encode_move(from, to, pc, (Movetype::bishopPromo | movetype)), list);
        AddMove(encode_move(from, to, pc, (Movetype::knightPromo | movetype)), list);
    }
    else { // else do not include possible promotions
        AddMove(encode_move(from, to, pc,  movetype), list);
    }
}

template<bool genQuiet, bool genCaptures>
static inline Bitboard LegalPawnMoves(S_Board* pos, int color, int square) {
    const Bitboard enemy = pos->Enemy();

    static_assert(genQuiet || genCaptures);

    // If we are pinned diagonally we can only do captures which are on the pin_dg
    // and on the checkmask
    if (pos->pinD & (1ULL << square))
        return genCaptures ? pawn_attacks[color][square] & pos->pinD & pos->checkMask & (enemy | (GetEpSquare(pos) != no_sq ? 1ULL << GetEpSquare(pos) : 0))
                           : 0ULL;

    // Calculate pawn pushs
    Bitboard push = PawnPush(color, square) & ~pos->Occupancy(BOTH);
    if constexpr (genCaptures && !genQuiet)
        push &= 0xFF000000000000FFULL;
    else {
        push |=
            (color == WHITE)
            ? (get_rank[square] == 1 ? (push >> 8) & ~pos->Occupancy(BOTH) : 0ULL)
            : (get_rank[square] == 6 ? (push << 8) & ~pos->Occupancy(BOTH) : 0ULL);
        if constexpr (genQuiet && !genCaptures)
            push &= ~0xFF000000000000FFULL;
    }

    // If we are pinned horizontally we can do no moves but if we are pinned
    // vertically we can only do pawn pushs
    if (pos->pinHV & (1ULL << square))
        return push & pos->pinHV & pos->checkMask;

    int offset = color * -16 + 8;
    Bitboard attacks = genCaptures ? pawn_attacks[color][square] : 0ULL;

    // If we are in check and  the en passant square lies on our attackmask and
    // the en passant piece gives check return the ep mask as a move square
    if (pos->checkMask != 18446744073709551615ULL && GetEpSquare(pos) != no_sq &&
        attacks & (1ULL << GetEpSquare(pos)) &&
        pos->checkMask & (1ULL << (GetEpSquare(pos) + offset)))
        return genCaptures ? 1ULL << GetEpSquare(pos) : 0ULL;

    // If we are in check we can do all moves that are on the checkmask
    if (pos->checkMask != 18446744073709551615ULL)
        return ((attacks & enemy) | push) & pos->checkMask;

    Bitboard moves = ((attacks & enemy) | push) & pos->checkMask;

    if (genCaptures && GetEpSquare(pos) != no_sq && SquareDistance(square, GetEpSquare(pos)) == 1 &&
        (1ULL << GetEpSquare(pos)) & attacks) {
        int ourPawn = GetPiece(PAWN, color);
        int theirPawn = GetPiece(PAWN, color ^ 1);
        int kSQ = KingSQ(pos, color);
        ClearPiece(ourPawn, square, pos);
        ClearPiece(theirPawn, (GetEpSquare(pos) + offset), pos);
        AddPiece(ourPawn, GetEpSquare(pos), pos);
        if (!((GetRookAttacks(kSQ, pos->Occupancy(BOTH)) &
            (pos->GetPieceColorBB(ROOK, color ^ 1) |
                pos->GetPieceColorBB(QUEEN, color ^ 1)))))
            moves |= (1ULL << GetEpSquare(pos));
        AddPiece(ourPawn, square, pos);
        AddPiece(theirPawn, GetEpSquare(pos) + offset, pos);
        ClearPiece(ourPawn, GetEpSquare(pos), pos);
    }

    return moves;
}

static inline Bitboard LegalKnightMoves(S_Board* pos, int color, int square) {
    if ((pos->pinD | pos->pinHV) & (1ULL << square))
        return NOMOVE;
    return knight_attacks[square] & ~pos->Occupancy(color) &
        pos->checkMask;
}

static inline Bitboard LegalBishopMoves(S_Board* pos, int color, int square) {
    if (pos->pinHV & (1ULL << square))
        return NOMOVE;
    if (pos->pinD & (1ULL << square))
        return GetBishopAttacks(square, pos->Occupancy(BOTH)) &
        ~(pos->Occupancy(color)) & pos->pinD & pos->checkMask;
    return GetBishopAttacks(square, pos->Occupancy(BOTH)) &
        ~(pos->Occupancy(color)) & pos->checkMask;
}

static inline Bitboard LegalRookMoves(S_Board* pos, int color, int square) {
    if (pos->pinD & (1ULL << square))
        return NOMOVE;
    if (pos->pinHV & (1ULL << square))
        return GetRookAttacks(square, pos->Occupancy(BOTH)) &
        ~(pos->Occupancy(color)) & pos->pinHV & pos->checkMask;
    return GetRookAttacks(square, pos->Occupancy(BOTH)) &
        ~(pos->Occupancy(color)) & pos->checkMask;
}

static inline Bitboard LegalQueenMoves(S_Board* pos, int color, int square) {
    return LegalRookMoves(pos, color, square) |
        LegalBishopMoves(pos, color, square);
}

static inline Bitboard LegalKingMoves(S_Board* pos, int color, int square) {
    Bitboard moves = king_attacks[square] & ~pos->Occupancy(color);
    Bitboard finalMoves = NOMOVE;
    int king = GetPiece(KING, color);
    ClearPiece(king, square, pos);
    while (moves) {
        int index = GetLsbIndex(moves);
        pop_bit(moves, index);
        if (IsSquareAttacked(pos, index, pos->side ^ 1)) {
            continue;
        }
        finalMoves |= (1ULL << index);
    }
    AddPiece(king, square, pos);

    return finalMoves;
}

// generate all moves
void GenerateMoves(S_MOVELIST* move_list, S_Board* pos) { // init move count
    move_list->count = 0;

    // define source & target squares
    int sourceSquare, targetSquare;

    init(pos, pos->side, KingSQ(pos, pos->side));

    if (pos->checks < 2) {
        Bitboard pawns = pos->GetPieceColorBB(PAWN, pos->side);
        while (pawns) {
            // init source square
            sourceSquare = GetLsbIndex(pawns);
            Bitboard moves = LegalPawnMoves<true, true>(pos, pos->side, sourceSquare);
            while (moves) {
                // init target square
                targetSquare = GetLsbIndex(moves);
                AddPawnMove(pos, sourceSquare, targetSquare, move_list);
                pop_bit(moves, targetSquare);
            }
            // pop lsb from piece bitboard copy
            pop_bit(pawns, sourceSquare);
        }

        // genarate knight moves
        Bitboard knights = pos->GetPieceColorBB(KNIGHT, pos->side);
        while (knights) {
            sourceSquare = GetLsbIndex(knights);
            Bitboard moves = LegalKnightMoves(pos, pos->side, sourceSquare);
            const int piece = GetPiece(KNIGHT, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                Movetype movetype = pos->PieceOn(targetSquare) != EMPTY ? Movetype::Capture : Movetype::Quiet;
                AddMove(encode_move(sourceSquare, targetSquare, piece, movetype), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(knights, sourceSquare);
        }

        Bitboard bishops = pos->GetPieceColorBB(BISHOP, pos->side);
        while (bishops) {
            sourceSquare = GetLsbIndex(bishops);
            Bitboard moves = LegalBishopMoves(pos, pos->side, sourceSquare);
            const int piece = GetPiece(BISHOP, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                Movetype movetype = pos->PieceOn(targetSquare) != EMPTY ? Movetype::Capture : Movetype::Quiet;
                AddMove(encode_move(sourceSquare, targetSquare, piece, movetype), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(bishops, sourceSquare);
        }

        Bitboard rooks = pos->GetPieceColorBB(ROOK, pos->side);
        while (rooks) {
            sourceSquare = GetLsbIndex(rooks);
            Bitboard moves = LegalRookMoves(pos, pos->side, sourceSquare);
            const int piece = GetPiece(ROOK, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                Movetype movetype = pos->PieceOn(targetSquare) != EMPTY ? Movetype::Capture : Movetype::Quiet;
                AddMove(encode_move(sourceSquare, targetSquare, piece, movetype), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(rooks, sourceSquare);
        }

        Bitboard queens = pos->GetPieceColorBB(QUEEN, pos->side);
        while (queens) {
            sourceSquare = GetLsbIndex(queens);
            Bitboard moves = LegalQueenMoves(pos, pos->side, sourceSquare);
            const int piece = GetPiece(QUEEN, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                Movetype movetype = pos->PieceOn(targetSquare) != EMPTY ? Movetype::Capture : Movetype::Quiet;
                AddMove(encode_move(sourceSquare, targetSquare, piece, movetype), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(queens, sourceSquare);
        }
    }

    sourceSquare = KingSQ(pos, pos->side);
    const int piece = GetPiece(KING, pos->side);
    Bitboard moves = LegalKingMoves(pos, pos->side, sourceSquare);
    while (moves) {
        targetSquare = GetLsbIndex(moves);
        Movetype movetype = pos->PieceOn(targetSquare) != EMPTY ? Movetype::Capture : Movetype::Quiet;
        AddMove(encode_move(sourceSquare, targetSquare, piece, movetype), move_list);
        pop_bit(moves, targetSquare);
    }

    if (pos->checkMask == 18446744073709551615ULL) {
        if (pos->side == WHITE) {
            // king side castling is available
            if (pos->GetCastlingPerm() & WKCA) {
                // make sure square between king and king's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), f1) &&
                    !get_bit(pos->Occupancy(BOTH), g1)) {
                    // make sure king and the f1 squares are not under attacks
                    if (!IsSquareAttacked(pos, e1, BLACK) &&
                        !IsSquareAttacked(pos, f1, BLACK) &&
                        !IsSquareAttacked(pos, g1, BLACK))
                        AddMove(encode_move(e1, g1, WK, Movetype::KSCastle), move_list);
                }
            }

            if (pos->GetCastlingPerm() & WQCA) {
                // make sure square between king and queen's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), d1) &&
                    !get_bit(pos->Occupancy(BOTH), c1) &&
                    !get_bit(pos->Occupancy(BOTH), b1)) {
                    // make sure king and the d1 squares are not under attacks
                    if (!IsSquareAttacked(pos, e1, BLACK) &&
                        !IsSquareAttacked(pos, d1, BLACK) &&
                        !IsSquareAttacked(pos, c1, BLACK))
                        AddMove(encode_move(e1, c1, WK, Movetype::QSCastle), move_list);
                }
            }
        }

        else {
            if (pos->GetCastlingPerm() & BKCA) {
                // make sure square between king and king's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), f8) &&
                    !get_bit(pos->Occupancy(BOTH), g8)) {
                    // make sure king and the f8 squares are not under attacks
                    if (!IsSquareAttacked(pos, e8, WHITE) &&
                        !IsSquareAttacked(pos, f8, WHITE) &&
                        !IsSquareAttacked(pos, g8, WHITE))
                        AddMove(encode_move(e8, g8, BK, Movetype::KSCastle), move_list);
                }
            }

            if (pos->GetCastlingPerm() & BQCA) {
                // make sure square between king and queen's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), d8) &&
                    !get_bit(pos->Occupancy(BOTH), c8) &&
                    !get_bit(pos->Occupancy(BOTH), b8)) {
                    // make sure king and the d8 squares are not under attacks
                    if (!IsSquareAttacked(pos, e8, WHITE) &&
                        !IsSquareAttacked(pos, d8, WHITE) &&
                        !IsSquareAttacked(pos, c8, WHITE))
                        AddMove(encode_move(e8, c8, BK, Movetype::QSCastle), move_list);
                }
            }
        }
    }
}

// generate all captures
void GenerateCaptures(S_MOVELIST* move_list, S_Board* pos) {
    // init move count
    move_list->count = 0;

    // define source & target squares
    int sourceSquare, targetSquare;

    if (pos->checks < 2) {
        Bitboard pawn_mask = pos->GetPieceColorBB(PAWN, pos->side);
        Bitboard knights_mask = pos->GetPieceColorBB(KNIGHT, pos->side);
        Bitboard bishops_mask = pos->GetPieceColorBB(BISHOP, pos->side);
        Bitboard rooks_mask = pos->GetPieceColorBB(ROOK, pos->side);
        Bitboard queens_mask = pos->GetPieceColorBB(QUEEN, pos->side);

        while (pawn_mask) {
            // init source square
            sourceSquare = GetLsbIndex(pawn_mask);
            Bitboard moves = LegalPawnMoves<false, true>(pos, pos->side, sourceSquare);

            while (moves) {
                // init target square
                targetSquare = GetLsbIndex(moves);
                AddPawnMove(pos, sourceSquare, targetSquare, move_list);
                pop_bit(moves, targetSquare);
            }

            // pop lsb from piece bitboard copy
            pop_bit(pawn_mask, sourceSquare);
        }
        // genarate knight moves
        while (knights_mask) {
            sourceSquare = GetLsbIndex(knights_mask);
            Bitboard moves = LegalKnightMoves(pos, pos->side, sourceSquare) & pos->Enemy();
            const int piece = GetPiece(KNIGHT, pos->side);
            // while we have moves that the knight can play we add them to the list
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Capture), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(knights_mask, sourceSquare);
        }

        while (bishops_mask) {
            sourceSquare = GetLsbIndex(bishops_mask);
            Bitboard moves = LegalBishopMoves(pos, pos->side, sourceSquare) & pos->Enemy();
            const int piece = GetPiece(BISHOP, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Capture), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(bishops_mask, sourceSquare);
        }

        while (rooks_mask) {
            sourceSquare = GetLsbIndex(rooks_mask);
            Bitboard moves = LegalRookMoves(pos, pos->side, sourceSquare) & pos->Enemy();
            const int piece = GetPiece(ROOK, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Capture), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(rooks_mask, sourceSquare);
        }

        while (queens_mask) {
            sourceSquare = GetLsbIndex(queens_mask);
            Bitboard moves = LegalQueenMoves(pos, pos->side, sourceSquare) & pos->Enemy();
            const int piece = GetPiece(QUEEN, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Capture), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(queens_mask, sourceSquare);
        }
    }

    sourceSquare = KingSQ(pos, pos->side);
    const int piece = GetPiece(KING, pos->side);
    Bitboard king_moves = LegalKingMoves(pos, pos->side, sourceSquare) & pos->Enemy();

    while (king_moves) {
        targetSquare = GetLsbIndex(king_moves);
        pop_bit(king_moves, targetSquare);
        AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Capture), move_list);
    }
}

// Generate all quiet moves
void GenerateQuiets(S_MOVELIST* move_list, S_Board* pos) {
    // init move count
    move_list->count = 0;

    // define source & target squares
    int sourceSquare, targetSquare;

    if (pos->checks < 2) {
        Bitboard pawn_mask = pos->GetPieceColorBB(PAWN, pos->side);
        Bitboard knights_mask = pos->GetPieceColorBB(KNIGHT, pos->side);
        Bitboard bishops_mask = pos->GetPieceColorBB(BISHOP, pos->side);
        Bitboard rooks_mask = pos->GetPieceColorBB(ROOK, pos->side);
        Bitboard queens_mask = pos->GetPieceColorBB(QUEEN, pos->side);

        while (pawn_mask) {
            // init source square
            sourceSquare = GetLsbIndex(pawn_mask);
            Bitboard moves = LegalPawnMoves<true, false>(pos, pos->side, sourceSquare);
            while (moves) {
                // init target square
                targetSquare = GetLsbIndex(moves);
                AddPawnMove(pos, sourceSquare, targetSquare, move_list);
                pop_bit(moves, targetSquare);
            }

            // pop lsb from piece bitboard copy
            pop_bit(pawn_mask, sourceSquare);
        }
        // genarate knight moves
        while (knights_mask) {
            sourceSquare = GetLsbIndex(knights_mask);
            Bitboard moves = LegalKnightMoves(pos, pos->side, sourceSquare) & ~pos->Enemy();
            const int piece = GetPiece(KNIGHT, pos->side);
            // while we have moves that the knight can play we add them to the list
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Quiet), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(knights_mask, sourceSquare);
        }

        while (bishops_mask) {
            sourceSquare = GetLsbIndex(bishops_mask);
            Bitboard moves = LegalBishopMoves(pos, pos->side, sourceSquare) & ~pos->Enemy();
            const int piece = GetPiece(BISHOP, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Quiet), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(bishops_mask, sourceSquare);
        }

        while (rooks_mask) {
            sourceSquare = GetLsbIndex(rooks_mask);
            Bitboard moves = LegalRookMoves(pos, pos->side, sourceSquare) & ~pos->Enemy();
            const int piece = GetPiece(ROOK, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Quiet), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(rooks_mask, sourceSquare);
        }

        while (queens_mask) {
            sourceSquare = GetLsbIndex(queens_mask);
            Bitboard moves = LegalQueenMoves(pos, pos->side, sourceSquare) & ~pos->Enemy();
            const int piece = GetPiece(QUEEN, pos->side);
            while (moves) {
                targetSquare = GetLsbIndex(moves);
                AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Quiet), move_list);
                pop_bit(moves, targetSquare);
            }
            pop_bit(queens_mask, sourceSquare);
        }
    }

    sourceSquare = KingSQ(pos, pos->side);
    const int piece = GetPiece(KING, pos->side);
    Bitboard king_moves = LegalKingMoves(pos, pos->side, sourceSquare) & ~pos->Enemy();

    while (king_moves) {
        targetSquare = GetLsbIndex(king_moves);
        pop_bit(king_moves, targetSquare);
        AddMove(encode_move(sourceSquare, targetSquare, piece, Movetype::Quiet), move_list);
    }

    if (pos->checkMask == 18446744073709551615ULL) {
        if (pos->side == WHITE) {
            // king side castling is available
            if (pos->GetCastlingPerm() & WKCA) {
                // make sure square between king and king's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), f1) &&
                    !get_bit(pos->Occupancy(BOTH), g1)) {
                    // make sure king and the f1 squares are not under attacks
                    if (!IsSquareAttacked(pos, e1, BLACK) &&
                        !IsSquareAttacked(pos, f1, BLACK) &&
                        !IsSquareAttacked(pos, g1, BLACK))
                        AddMove(encode_move(e1, g1, WK, Movetype::KSCastle), move_list);
                }
            }

            if (pos->GetCastlingPerm() & WQCA) {
                // make sure square between king and queen's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), d1) &&
                    !get_bit(pos->Occupancy(BOTH), c1) &&
                    !get_bit(pos->Occupancy(BOTH), b1)) {
                    // make sure king and the d1 squares are not under attacks
                    if (!IsSquareAttacked(pos, e1, BLACK) &&
                        !IsSquareAttacked(pos, d1, BLACK) &&
                        !IsSquareAttacked(pos, c1, BLACK))
                        AddMove(encode_move(e1, c1, WK, Movetype::QSCastle), move_list);
                }
            }
        }

        else {
            if (pos->GetCastlingPerm() & BKCA) {
                // make sure square between king and king's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), f8) &&
                    !get_bit(pos->Occupancy(BOTH), g8)) {
                    // make sure king and the f8 squares are not under attacks
                    if (!IsSquareAttacked(pos, e8, WHITE) &&
                        !IsSquareAttacked(pos, f8, WHITE) &&
                        !IsSquareAttacked(pos, g8, WHITE))
                        AddMove(encode_move(e8, g8, BK, Movetype::KSCastle), move_list);
                }
            }

            if (pos->GetCastlingPerm() & BQCA) {
                // make sure square between king and queen's rook are empty
                if (!get_bit(pos->Occupancy(BOTH), d8) &&
                    !get_bit(pos->Occupancy(BOTH), c8) &&
                    !get_bit(pos->Occupancy(BOTH), b8)) {
                    // make sure king and the d8 squares are not under attacks
                    if (!IsSquareAttacked(pos, e8, WHITE) &&
                        !IsSquareAttacked(pos, d8, WHITE) &&
                        !IsSquareAttacked(pos, c8, WHITE))
                        AddMove(encode_move(e8, c8, BK, Movetype::QSCastle), move_list);
                }
            }
        }
    }
}

bool MoveIsLegal(S_Board* pos, const int move) {

    if (move == NOMOVE)
        return false;

    int sourceSquare = From(move);
    int toSquare = To(move);
    int piece = pos->PieceOn(sourceSquare);
    if (piece == EMPTY)
        return false;

    if (piece != Piece(move))
        return false;

    if (piece / 6 != pos->side)
        return false;

    if (GetPieceType(pos->PieceOn(toSquare)) == KING)
        return false;

    if (IsCapture(move) && (pos->PieceOn(toSquare) == EMPTY))
        return false;

    int pieceType = GetPieceType(piece);

    if ((isDP(move) || isPromo(move) || isEnpassant(move)) && pieceType != PAWN)
        return false;

    if (isPromo(move) && !((1ULL << toSquare) & 0xFF000000000000FFULL))
        return false;

    if ((IsCastle(move) || pos->checks >= 2) && pieceType != KING)
        return false;

    if (pieceType == PAWN && (toSquare == GetEpSquare(pos)) != isEnpassant(move))
        return false;

    if ((pos->PieceOn(toSquare) == EMPTY && !isEnpassant(move) && !isPromo(move)) != IsQuiet(move))
        return false;

    Bitboard legalMoves = 0;
    if (pieceType == PAWN)
        legalMoves = IsQuiet(move) ? LegalPawnMoves<true, false>(pos, pos->side, sourceSquare)
                                   : LegalPawnMoves<false, true>(pos, pos->side, sourceSquare);
    else if (pieceType == KNIGHT)
        legalMoves = LegalKnightMoves(pos, pos->side, sourceSquare);
    else if (pieceType == BISHOP)
        legalMoves = LegalBishopMoves(pos, pos->side, sourceSquare);
    else if (pieceType == ROOK)
        legalMoves = LegalRookMoves(pos, pos->side, sourceSquare);
    else if (pieceType == QUEEN)
        legalMoves = LegalQueenMoves(pos, pos->side, sourceSquare);

    // pieceType == KING
    else {
        if (IsCastle(move)) {
            if (pos->checkMask != 18446744073709551615ULL)
                return false;

            if (pos->side == WHITE) {
                if (Movetype(GetMovetype(move)) == Movetype::KSCastle)
                    return (    pos->GetCastlingPerm() & WKCA 
                            && !get_bit(pos->Occupancy(BOTH), f1)
                            && !get_bit(pos->Occupancy(BOTH), g1)
                            && !IsSquareAttacked(pos, e1, BLACK)
                            && !IsSquareAttacked(pos, f1, BLACK)
                            && !IsSquareAttacked(pos, g1, BLACK));

                if (Movetype(GetMovetype(move)) == Movetype::QSCastle)
                    return (    pos->GetCastlingPerm() & WQCA 
                            && !get_bit(pos->Occupancy(BOTH), d1)
                            && !get_bit(pos->Occupancy(BOTH), c1)
                            && !get_bit(pos->Occupancy(BOTH), b1)
                            && !IsSquareAttacked(pos, e1, BLACK)
                            && !IsSquareAttacked(pos, d1, BLACK)
                            && !IsSquareAttacked(pos, c1, BLACK));
            } else {
                if (Movetype(GetMovetype(move)) == Movetype::KSCastle)
                    return (    pos->GetCastlingPerm() & BKCA 
                            && !get_bit(pos->Occupancy(BOTH), f8)
                            && !get_bit(pos->Occupancy(BOTH), g8)
                            && !IsSquareAttacked(pos, e8, WHITE)
                            && !IsSquareAttacked(pos, f8, WHITE)
                            && !IsSquareAttacked(pos, g8, WHITE));

                if (Movetype(GetMovetype(move)) == Movetype::QSCastle)
                    return (    pos->GetCastlingPerm() & BQCA 
                            && !get_bit(pos->Occupancy(BOTH), d8)
                            && !get_bit(pos->Occupancy(BOTH), c8)
                            && !get_bit(pos->Occupancy(BOTH), b8)
                            && !IsSquareAttacked(pos, e8, WHITE)
                            && !IsSquareAttacked(pos, d8, WHITE)
                            && !IsSquareAttacked(pos, c8, WHITE));
            }

            return false;
        }
        else
            legalMoves = LegalKingMoves(pos, pos->side, sourceSquare);
    }

    return legalMoves & (1ULL << toSquare);
}