#pragma once

#include <cstdint>
#include <array>
#include <vector>

#if defined(USE_AVX512) || defined(USE_AVX2)
#include <immintrin.h>
#endif

// Net Arch: (INPUT_SIZE -> L1_SIZE) x 2 -> 1 x OUTPUT_BUCKETS
constexpr int INPUT_SIZE = 768;
constexpr int L1_SIZE = 1536;
constexpr int OUTPUT_BUCKETS = 8;

constexpr int FT_QUANT = 181;
constexpr int L1_QUANT = 256;
constexpr int NET_SCALE = 400;

#if defined(USE_AVX512)
constexpr int L1_CHUNK_SIZE = sizeof(__m512i) / sizeof(int16_t);
#elif defined(USE_AVX2)
constexpr int L1_CHUNK_SIZE = sizeof(__m256i) / sizeof(int16_t);
#else
constexpr int L1_CHUNK_SIZE = 1;
#endif

using NNUEIndices = std::pair<std::size_t, std::size_t>;

struct Network {
    int16_t FTWeights[INPUT_SIZE * L1_SIZE];
    int16_t FTBiases[L1_SIZE];
    int16_t L1Weights[OUTPUT_BUCKETS][2 * L1_SIZE];
    int32_t L1Biases[OUTPUT_BUCKETS];
};

struct UnquantisedNetwork {
    float FTWeights[INPUT_SIZE * L1_SIZE];
    float FTBiases[L1_SIZE];
    float L1Weights[2 * L1_SIZE][OUTPUT_BUCKETS];
    float L1Biases[OUTPUT_BUCKETS];
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
    void ActivateFTAndAffineL1(const int16_t *inputs, const int16_t *weights, const int32_t bias, int &output);
    [[nodiscard]] int output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket);
    [[nodiscard]] NNUEIndices GetIndex(const int piece, const int square);
    #if defined(USE_AVX2)
    [[nodiscard]] int hadd_int32(const __m256i sum);
    #endif
};
