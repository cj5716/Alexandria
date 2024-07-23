#include "history.h"
#include <algorithm>
#include <cstring>
#include "position.h"
#include "move.h"
#include "search.h"

int16_t HistoryBonus(const int depth) {
    return std::min(histBonusQuadratic() * depth * depth + histBonusLinear() * depth + histBonusConst(), histBonusMax());
}

// Quiet history is a history table indexed by [side-to-move][from-sq-is-attacked][from-to-of-move].
void UpdateQuietHistory(const Position *pos, SearchData *sd, const Move move, const int16_t bonus) {

    int16_t &entry = sd->quietHistory[pos->side][IsAttackedByOpp(pos, From(move))][FromTo(move)];

    // Scale the bonus so that the history, when updated, will be within [-quietHistMax(), quietHistMax()]
    const int scaledBonus = bonus - entry * std::abs(bonus) / quietHistMax();
    entry += bonus;
}

int16_t GetQuietHistoryScore(const Position *pos, const SearchData *sd, const Move move) {
    return sd->quietHistory[pos->side][IsAttackedByOpp(pos, From(move))][FromTo(move)];
}

// Tactical history is a history table indexed by [piece-to-of-move][captured-piecetype].
void UpdateTacticalHistory(const Position *pos, SearchData *sd, const Move move, const int16_t bonus) {

    int capturedPieceType = GetPieceType(pos->PieceOn(To(move)));
    if (capturedPieceType == EMPTY) capturedPieceType = KING; // As it is impossible to capture a king, we use the "King" slot for "Empty"
    int16_t &entry = sd->tacticalHistory[PieceTo(move)][capturedPieceType];

    // Scale the bonus so that the history, when updated, will be within [-tacticalHistMax(), tacticalHistMax()]
    const int scaledBonus = bonus - entry * std::abs(bonus) / tacticalHistMax();
    entry += bonus;
}

int16_t GetTacticalHistoryScore(const Position *pos, const SearchData *sd, const Move move) {

    int capturedPieceType = GetPieceType(pos->PieceOn(To(move)));
    if (capturedPieceType == EMPTY) capturedPieceType = KING; // As it is impossible to capture a king, we use the "King" slot for "Empty"

    return sd->tacticalHistory[PieceTo(move)][capturedPieceType];
}

// Use this function to update all quiet histories
void UpdateAllHistories(const Position *pos, const SearchStack *ss, SearchData *sd, const int depth, const Move bestMove, const MoveList &quietMoves, const MoveList &tacticalMoves) {
    int16_t bonus = HistoryBonus(depth);
    if (isTactical(bestMove)) {
        // Positively update the move that failed high
        UpdateTacticalHistory(pos, sd, bestMove, bonus);
    }
    else {
        // Positively update the move that failed high
        UpdateQuietHistory(pos, sd, bestMove, bonus);

        // Penalise all quiets that failed to do so (they were ordered earlier but weren't as good)
        for (int i = 0; i < quietMoves.count; ++i) {
            Move quiet = quietMoves.moves[i].move;
            if (bestMove == quiet) continue;
            UpdateQuietHistory(pos, sd, quiet, -bonus);
        }
    }

    // Penalise all captures that have failed to do so, even if the best move was quiet
    for (int i = 0; i < tacticalMoves.count; ++i) {
        Move tactical = tacticalMoves.moves[i].move;
        if (bestMove == tactical) continue;
        UpdateTacticalHistory(pos, sd, tactical, -bonus);
    }
}

int GetHistoryScore(const Position *pos, const SearchStack *ss, const SearchData *sd, const Move move) {
    if (isTactical(move)) {
        return GetTacticalHistoryScore(pos, sd, move);
    }
    else {
        return GetQuietHistoryScore(pos, sd, move);
    }
}

// Resets the history tables
void CleanHistories(SearchData *sd) {
    std::memset(sd->quietHistory, 0, sizeof(sd->quietHistory));
}
