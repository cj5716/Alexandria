#pragma once

#include "board.h"
#include "types.h"
#include <vector>

constexpr int ENTRIES_PER_BUCKET = 5;

// 12 bytes:
// 2 for move
// 2 for score
// 2 for eval
// 4 for key
// 1 for depth
// 1 for age + bound + PV
PACK(struct S_HashEntry {
    int16_t move = NOMOVE;
    int16_t score = SCORE_NONE;
    int16_t eval = SCORE_NONE;
    TTKey ttKey = 0;
    uint8_t depth = 0;
    uint8_t ageBoundPV = HFNONE; // lower 2 bits is bound, 3rd bit is PV, next 5 is age
});

// Packs the 12-byte entries into 64-byte buckets
// 5 entries per bucket with 4 bytes of padding
typedef struct {
    S_HashEntry entries[ENTRIES_PER_BUCKET];
    uint32_t padding;
} S_HashBucket;

static_assert(sizeof(S_HashBucket) == 64);

struct S_HashTable {
    std::vector<S_HashBucket> pTable;
    uint8_t age;
};

extern S_HashTable HashTable[1];

constexpr uint8_t MAX_AGE = 1 << 5; // must be power of 2
constexpr uint8_t AGE_MASK = MAX_AGE - 1;

void ClearHashTable(S_HashTable* table);
// Initialize an Hashtable of size MB
void InitHashTable(S_HashTable* table, uint64_t MB);

[[nodiscard]] bool ProbeHashEntry(const ZobristKey posKey, S_HashEntry* tte);

void StoreHashEntry(const ZobristKey key, const int16_t move, int score, int16_t eval, const int bound, const int depth, const bool pv, const bool wasPV);

[[nodiscard]] uint64_t Index(const ZobristKey posKey);

int GetHashfull();

void TTPrefetch(const ZobristKey posKey);

int ScoreToTT(int score, int ply);

int ScoreFromTT(int score, int ply);

int16_t MoveToTT(int move);

int MoveFromTT(S_Board *pos, int16_t packed_move);

uint8_t BoundFromTT(uint8_t ageBoundPV);

bool FormerPV(uint8_t ageBoundPV);

uint8_t AgeFromTT(uint8_t ageBoundPV);

uint8_t PackToTT(uint8_t bound, bool wasPV, uint8_t age);

void UpdateTableAge();