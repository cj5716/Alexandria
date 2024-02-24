#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <immintrin.h>

constexpr int INPUT_WEIGHTS = 768;
constexpr int HIDDEN_SIZE = 1024;
constexpr int OUTPUT_BUCKETS = 1;

struct Network {
    int16_t featureWeights[INPUT_WEIGHTS * HIDDEN_SIZE];
    int16_t featureBias[HIDDEN_SIZE];
    int16_t outputWeights[HIDDEN_SIZE * 2 * OUTPUT_BUCKETS];
    int16_t outputBias[OUTPUT_BUCKETS];
};

extern Network net;

class NNUE {
public:
    using accumulator = std::array<std::array<int16_t, HIDDEN_SIZE>, 2>;

    void init(const char* file);
    void add(NNUE::accumulator& board_accumulator, const int piece, const int to);
    void update(NNUE::accumulator& board_accumulator, std::vector<std::pair<std::size_t, std::size_t>>& NNUEAdd, std::vector<std::pair<std::size_t, std::size_t>>& NNUESub);
    void addSub(NNUE::accumulator& board_accumulator, std::size_t whiteAddIdx, std::size_t blackAddIdx, std::size_t whiteSubIdx, std::size_t blackSubIdx);
    void addSubSub(NNUE::accumulator& board_accumulator, std::size_t whiteAddIdx, std::size_t blackAddIdx, std::size_t whiteSubIdx1, std::size_t blackSubIdx1, std::size_t whiteSubIdx2, std::size_t blackSubIdx2);
    [[nodiscard]] int32_t flatten(const int16_t *acc, const int16_t *weights, const int outputBucket);
    [[nodiscard]] int32_t output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket);
    [[nodiscard]] std::pair<std::size_t, std::size_t> GetIndex(const int piece, const int square);
    #if defined(USE_AVX2)
    [[nodiscard]] int32_t horizontal_add(const __m256i sum);
    #endif
};
