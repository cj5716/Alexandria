#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "types.h"

constexpr int NUM_FEATURE_INPUTS = 768;
constexpr int NUM_FEATURES = 32;

using FeatureIndices = std::pair<std::size_t, std::size_t>;

struct FeatureNet {
    int16_t FeatureWeights[NUM_FEATURE_INPUTS * NUM_FEATURES];
    int16_t FeatureBiases [NUM_FEATURES];
    void init(const char *file);
};

extern FeatureNet featureNet;
struct Position;

struct FeatureAccumulator {
    std::array<std::array<int16_t, NUM_FEATURES>, 2> values;
    std::vector<FeatureIndices> FeatureAdd = {};
    std::vector<FeatureIndices> FeatureSub = {};

    FeatureIndices GetIndex(const int piece, const int square) {
        constexpr std::size_t COLOR_STRIDE = 64 * 6;
        constexpr std::size_t PIECE_STRIDE = 64;
        int piecetype = PieceType[piece];
        int color = Color[piece];
        std::size_t whiteIdx = color * COLOR_STRIDE + piecetype * PIECE_STRIDE + (square ^ 0b111'000);
        std::size_t blackIdx = (1 ^ color) * COLOR_STRIDE + piecetype * PIECE_STRIDE + square;
        return {whiteIdx, blackIdx};
    }

    void AppendAddIndex(const int piece, const int square) {
        FeatureAdd.emplace_back(GetIndex(piece, square));
    }

    void AppendSubIndex(const int piece, const int square) {
        FeatureSub.emplace_back(GetIndex(piece, square));
    }

    void Accumulate(Position *pos);

    uint64_t GetFeatureHash(const int side);
};

void UpdateFeatureAccumulator(FeatureAccumulator *acc);
void FeatureAddSub(FeatureAccumulator *new_acc, FeatureAccumulator *prev_acc, FeatureIndices add, FeatureIndices sub);
void FeatureAddSubSub(FeatureAccumulator *new_acc, FeatureAccumulator *prev_acc, FeatureIndices add, FeatureIndices sub1, FeatureIndices sub2);
uint64_t GetFeatureHash(Position *pos);