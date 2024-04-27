#pragma once

#include <cstdint>
#include <array>
#include <vector>

#if defined(USE_AVX512) || defined(USE_AVX2)
#include <immintrin.h>
#endif

// Net Arch: (INPUT_SIZE -> L1_SIZE) x 2 -> (L2_SIZE -> L3_SIZE -> 1) x OUTPUT_BUCKETS
constexpr int INPUT_SIZE = 768;
constexpr int L1_SIZE = 1536;
constexpr int L2_SIZE = 8;
constexpr int L3_SIZE = 32;
constexpr int OUTPUT_BUCKETS = 8;

constexpr int FT_QUANT = 128;
constexpr int L1_QUANT = 512;
constexpr int NET_SCALE = 400;

#if defined(USE_AVX512)
constexpr int L1_CHUNK_SIZE = sizeof(__m512i) / sizeof(int16_t);
constexpr int L2_CHUNK_SIZE = sizeof(__m256) / sizeof(float);
constexpr int L3_CHUNK_SIZE = sizeof(__m256) / sizeof(float);
#elif defined(USE_AVX2)
constexpr int L1_CHUNK_SIZE = sizeof(__m256i) / sizeof(int16_t);
constexpr int L2_CHUNK_SIZE = sizeof(__m256) / sizeof(float);
constexpr int L3_CHUNK_SIZE = sizeof(__m256) / sizeof(float);
#else
constexpr int L1_CHUNK_SIZE = 1;
constexpr int L2_CHUNK_SIZE = 1;
constexpr int L3_CHUNK_SIZE = 1;
#endif

using NNUEIndices = std::pair<std::size_t, std::size_t>;

struct Network {
    int16_t FTWeights[INPUT_SIZE * L1_SIZE];
    int16_t FTBiases[L1_SIZE];
    int16_t L1Weights[OUTPUT_BUCKETS][2 * L1_SIZE * L2_SIZE];
    float   L1Biases[OUTPUT_BUCKETS][L2_SIZE];
    float   L2Weights[OUTPUT_BUCKETS][L2_SIZE * L3_SIZE];
    float   L2Biases[OUTPUT_BUCKETS][L3_SIZE];
    float   L3Weights[OUTPUT_BUCKETS][L3_SIZE];
    float   L3Biases[OUTPUT_BUCKETS];
};

struct UnquantisedNetwork {
    float FTWeights[INPUT_SIZE * L1_SIZE];
    float FTBiases[L1_SIZE];
    float L1Weights[2 * L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
    float L1Biases[OUTPUT_BUCKETS][L2_SIZE];
    float L2Weights[L2_SIZE][OUTPUT_BUCKETS][L3_SIZE];
    float L2Biases[OUTPUT_BUCKETS][L3_SIZE];
    float L3Weights[L3_SIZE][OUTPUT_BUCKETS];
    float L3Biases[OUTPUT_BUCKETS];
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
    void ActivateFTAndAffineL1(const int16_t *inputs, const int16_t *weights, int *output);
    void ActivateL1AndAffineL2(const float *inputs, const float *weights, const float *biases, float *output);
    void ActivateL2AndAffineL3(const float *inputs, const float *weights, const float bias, float &output);
    [[nodiscard]] int output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket);
    [[nodiscard]] NNUEIndices GetIndex(const int piece, const int square);
    #if defined(USE_AVX512) || defined(USE_AVX2)
    [[nodiscard]] float _mm256_reduce_add_ps(const __m256 sum);
    [[nodiscard]] __m256i combine_m256i(const __m256i in0, const __m256i in1);
    [[nodiscard]] __m256  combine_m256(const __m256 in0, const __m256 in1);
    [[nodiscard]] __m256  hadd_psx4(const __m256* in);
    #endif
    #if defined(USE_AVX512)
    [[nodiscard]] __m256i m512_to_m256(const __m512i in);
    [[nodiscard]] __m256i hadd_epi32x4(const __m512i* in);
    #elif defined(USE_AVX2)
    [[nodiscard]] __m256i hadd_epi32x4(const __m256i* in);
    #endif
};
