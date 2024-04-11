#pragma once

#include <cstdint>
#include <array>
#include <vector>

#if defined(USE_AVX512) || defined(USE_AVX2)
#include <immintrin.h>
#endif

// Net Arch: (INPUT_SIZE -> L1_SIZE) x 2 -> (L2_SIZE -> 1) x OUTPUT_BUCKETS
constexpr int INPUT_SIZE = 768;
constexpr int L1_SIZE = 1536;
constexpr int L2_SIZE = 8;
constexpr int OUTPUT_BUCKETS = 8;

constexpr int QUANT = 128;
constexpr int NET_SCALE = 400;

using NNUEIndices = std::pair<std::size_t, std::size_t>;

struct Network {
    int16_t FTWeights[INPUT_SIZE * L1_SIZE];
    int16_t FTBiases[L1_SIZE];
    int16_t L1Weights[OUTPUT_BUCKETS][2 * L1_SIZE * L2_SIZE];
    int16_t L1Biases[OUTPUT_BUCKETS][L2_SIZE];
    float   L2Weights[OUTPUT_BUCKETS][L2_SIZE];
    float   L2Biases[OUTPUT_BUCKETS];
};

struct UnquantisedNetwork {
    float FTWeights[INPUT_SIZE * L1_SIZE];
    float FTBiases[L1_SIZE];
    float L1Weights[2 * L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
    float L1Biases[OUTPUT_BUCKETS][L2_SIZE];
    float L2Weights[L2_SIZE][OUTPUT_BUCKETS];
    float L2Biases[OUTPUT_BUCKETS];
};

extern Network net;

class NNUE {
public:
    using accumulator = std::array<std::array<int16_t, L1_SIZE>, 2>;

    void init(const char* file);
    void add(NNUE::accumulator& board_accumulator, const int piece, const int to);
    void update(NNUE::accumulator& board_accumulator, std::vector<NNUEIndices>& NNUEAdd, std::vector<NNUEIndices>& NNUESub);
    void addSub(NNUE::accumulator& board_accumulator, NNUEIndices add, NNUEIndices sub);
    void addSubSub(NNUE::accumulator& board_accumulator, NNUEIndices add, NNUEIndices sub1, NNUEIndices sub2);
    [[nodiscard]] float ActivateFTAndAffineL1(const int16_t *inputs, const int16_t *weights, const int16_t bias);
    [[nodiscard]] float ActivateL1AndAffineL2(const float *inputs, const float *weights, const float bias);
    [[nodiscard]] int32_t output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket);
    [[nodiscard]] NNUEIndices GetIndex(const int piece, const int square);
    #if defined(USE_AVX2)
    [[nodiscard]] int32_t hadd_int32(const __m256i sum);
    #endif
};
