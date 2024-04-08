#include "nnue.h"
#include <algorithm>
#include "position.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include "incbin/incbin.h"

#if defined(USE_AVX512) || defined(USE_AVX2)
#include <immintrin.h>
#endif

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEVALData[];  // a pointer to the embedded data
//     const unsigned char *const gEVALEnd;     // a marker to the end
//     const unsigned int         gEVALSize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER)
INCBIN(EVAL, EVALFILE);
#else
const unsigned char gEVALData[1] = {};
const unsigned char* const gEVALEnd = &gEVALData[1];
const unsigned int gEVALSize = 1;
#endif

Network net;

// Thanks to Disservin for having me look at his code and Luecx for the
// invaluable help and the immense patience

void NNUE::init(const char* file) {

    // open the nn file
    FILE* nn = fopen(file, "rb");

    // Unquantised network
    UnquantisedNetwork unquantisedNet;

    // if it's not invalid read the config values from it
    if (nn) {
        // initialize an accumulator for every input of the second layer
        size_t read = 0;
        const size_t fileSize = sizeof(UnquantisedNetwork);
        const size_t objectsExpected = fileSize / sizeof(float);

        read += fread(unquantisedNet.FTWeights, sizeof(float), INPUT_SIZE * L1_SIZE, nn);
        read += fread(unquantisedNet.FTBiases, sizeof(float), L1_SIZE, nn);
        read += fread(unquantisedNet.L1Weights, sizeof(float), OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE, nn);
        read += fread(unquantisedNet.L1Biases, sizeof(float), OUTPUT_BUCKETS * L2_SIZE, nn);
        read += fread(unquantisedNet.L2Weights, sizeof(float), OUTPUT_BUCKETS * L2_SIZE, nn);
        read += fread(unquantisedNet.L2Biases, sizeof(float), OUTPUT_BUCKETS, nn);

        if (read != objectsExpected) {
            std::cout << "Error loading the net, aborting ";
            std::cout << "Expected " << objectsExpected << " floats, got " << read << "\n";
            exit(1);
        }

        // after reading the config we can close the file
        fclose(nn);

    } else {
        // if we don't find the nnue file we use the net embedded in the exe
        uint64_t memoryIndex = 0;
        std::memcpy(unquantisedNet.FTWeights, &gEVALData[memoryIndex], sizeof(float) * INPUT_SIZE * L1_SIZE);
        memoryIndex += sizeof(float) * INPUT_SIZE * L1_SIZE;
        std::memcpy(unquantisedNet.FTBiases, &gEVALData[memoryIndex], sizeof(float) * L1_SIZE);
        memoryIndex += sizeof(float) * L1_SIZE;

        std::memcpy(unquantisedNet.L1Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE;
        std::memcpy(unquantisedNet.L1Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L2_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L2_SIZE;

        std::memcpy(unquantisedNet.L2Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L2_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L2_SIZE;
        std::memcpy(unquantisedNet.L2Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS;
    }

    // Quantise FT Weights
    for (int i = 0; i < INPUT_SIZE * L1_SIZE; ++i)
        net.FTWeights[i] = static_cast<int16_t>(unquantisedNet.FTWeights[i] * FT_QUANT);

    // Quantise FT Biases
    for (int i = 0; i < L1_SIZE; ++i)
        net.FTBiases[i] = static_cast<int16_t>(unquantisedNet.FTBiases[i] * FT_QUANT);

    // Transpose L1 and L2 weights and biases
    for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {

        // Quantise L1 Weights
        for (int i = 0; i < 2 * L1_SIZE * L2_SIZE; ++i)
            net.L1Weights[bucket][i] = static_cast<int16_t>(unquantisedNet.L1Weights[i][bucket] * L1_QUANT);

        // Quantise L1 Biases
        for (int i = 0; i < L2_SIZE; ++i)
            net.L1Biases[bucket][i] = static_cast<int16_t>(unquantisedNet.L1Biases[i][bucket] * L1_QUANT);

        // Quantise L2 Weights
        for (int i = 0; i < L2_SIZE; ++i)
            net.L2Weights[bucket][i] = static_cast<int16_t>(unquantisedNet.L2Weights[i][bucket] * L2_QUANT);

        // Quantise L2 Biases
        net.L2Biases[bucket] = static_cast<int16_t>(unquantisedNet.L2Biases[bucket] * L2_QUANT);
    }

}

#if defined(USE_AVX512)
constexpr int32_t CHUNK_SIZE = sizeof(__m512i) / sizeof(int16_t);
#elif defined(USE_AVX2)
constexpr int32_t CHUNK_SIZE = sizeof(__m256i) / sizeof(int16_t);
#else
constexpr int32_t CHUNK_SIZE = 1;
#endif

void NNUE::add(NNUE::accumulator& board_accumulator, const int piece, const int to) {
    auto [whiteIdx, blackIdx] = GetIndex(piece, to);
    auto whiteAdd = &net.FTWeights[whiteIdx * L1_SIZE];
    auto blackAdd = &net.FTWeights[blackIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[0][i] += whiteAdd[i];
    }
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[1][i] += blackAdd[i];
    }
}

void NNUE::update(NNUE::accumulator& board_accumulator, std::vector<NNUEIndices>& NNUEAdd, std::vector<NNUEIndices>& NNUESub) {
    int adds = NNUEAdd.size();
    int subs = NNUESub.size();

    if (adds == 0 && subs == 0)
        return;

    // Quiets
    if (adds == 1 && subs == 1) {
        addSub(board_accumulator, NNUEAdd[0], NNUESub[0]);
    }
    // Captures
    else if (adds == 1 && subs == 2) {
        addSubSub(board_accumulator, NNUEAdd[0], NNUESub[0], NNUESub[1]);
    }
    // Castling
    else {
        addSub(board_accumulator, NNUEAdd[0], NNUESub[0]);
        addSub(board_accumulator, NNUEAdd[1], NNUESub[1]);
    }
    // Reset the add and sub vectors
    NNUEAdd.clear();
    NNUESub.clear();
}

void NNUE::addSub(NNUE::accumulator& board_accumulator, NNUEIndices add, NNUEIndices sub) {
    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx, blackSubIdx] = sub;
    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub = &net.FTWeights[whiteSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[0][i] = board_accumulator[0][i] - whiteSub[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub = &net.FTWeights[blackSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[1][i] = board_accumulator[1][i] - blackSub[i] + blackAdd[i];
    }
}

void NNUE::addSubSub(NNUE::accumulator& board_accumulator, NNUEIndices add, NNUEIndices sub1, NNUEIndices sub2) {

    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx1, blackSubIdx1] = sub1;
    auto [whiteSubIdx2, blackSubIdx2] = sub2;

    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub1 = &net.FTWeights[whiteSubIdx1 * L1_SIZE];
    auto whiteSub2 = &net.FTWeights[whiteSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[0][i] = board_accumulator[0][i] - whiteSub1[i] - whiteSub2[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub1 = &net.FTWeights[blackSubIdx1 * L1_SIZE];
    auto blackSub2 = &net.FTWeights[blackSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator[1][i] = board_accumulator[1][i] - blackSub1[i] - blackSub2[i] + blackAdd[i];
    }
}

#if defined(USE_AVX2) && !defined(USE_AVX512)
int32_t NNUE::horizontal_add(const __m256i sum) {
    auto upper_128 = _mm256_extracti128_si256(sum, 1);
    auto lower_128 = _mm256_castsi256_si128(sum);
    auto sum_128 = _mm_add_epi32(upper_128, lower_128);

    auto upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
    auto sum_64 = _mm_add_epi32(upper_64, sum_128);

    auto upper_32 = _mm_shuffle_epi32(sum_64, 1);
    auto sum_32 = _mm_add_epi32(upper_32, sum_64);

    return _mm_cvtsi128_si32(sum_32);
}
#endif

int32_t NNUE::flattenL1(const int16_t *us, const int16_t *them, const int16_t *us_weights, const int16_t *them_weights, const int16_t bias) {
    int32_t sum = 0;
    #if defined(USE_AVX512)
    auto vec_sum = _mm512_setzero_si512();
    for (int i = 0; i < L1_SIZE / CHUNK_SIZE; i++) {
        auto min = _mm512_set1_epi16(0);
        auto max = _mm512_set1_epi16(FT_QUANT);

        auto us_weights_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(us_weights + i * CHUNK_SIZE));
        auto us_vector = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(us + i * CHUNK_SIZE));
        auto us_clipped = _mm512_min_epi16(_mm512_max_epi16(us_vector, min), max);
        auto us_mul = _mm512_madd_epi16(_mm512_mullo_epi16(us_clipped, us_weights_vec), us_clipped);
        vec_sum = _mm512_add_epi32(vec_sum, us_mul);

        auto them_weights_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(them_weights + i * CHUNK_SIZE));
        auto them_vector = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(them + i * CHUNK_SIZE));
        auto them_clipped = _mm512_min_epi16(_mm512_max_epi16(them_vector, min), max);
        auto them_mul = _mm512_madd_epi16(_mm512_mullo_epi16(them_clipped, them_weights_vec), them_clipped);
        vec_sum = _mm512_add_epi32(vec_sum, them_mul);
    }
    sum = _mm512_reduce_add_epi32(vec_sum);
    #elif defined(USE_AVX2)
    auto vec_sum = _mm256_setzero_si256();
    for (int i = 0; i < L1_SIZE / CHUNK_SIZE; i++) {
        auto min = _mm256_set1_epi16(0);
        auto max = _mm256_set1_epi16(FT_QUANT);

        auto us_weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(us_weights + i * CHUNK_SIZE));
        auto us_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(us + i * CHUNK_SIZE));
        auto us_clipped = _mm256_min_epi16(_mm256_max_epi16(us_vector, min), max);
        auto us_mul = _mm256_madd_epi16(_mm256_mullo_epi16(us_clipped, us_weights_vec), us_clipped);
        vec_sum = _mm256_add_epi32(vec_sum, us_mul);

        auto them_weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(them_weights + i * CHUNK_SIZE));
        auto them_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(them + i * CHUNK_SIZE));
        auto them_clipped = _mm256_min_epi16(_mm256_max_epi16(them_vector, min), max);
        auto them_mul = _mm256_madd_epi16(_mm256_mullo_epi16(them_clipped, them_weights_vec), them_clipped);
        vec_sum = _mm256_add_epi32(vec_sum, them_mul);
    }
    sum = horizontal_add(vec_sum);
    #else
    for (int i = 0; i < L1_SIZE; i++) {
        int32_t clipped = std::clamp(static_cast<int32_t>(us[i]), 0, FT_QUANT);
        sum += clipped * clipped * static_cast<int32_t>(us_weights[i]);

        int32_t clipped = std::clamp(static_cast<int32_t>(them[i]), 0, FT_QUANT);
        sum += clipped * clipped * static_cast<int32_t>(them_weights[i]);
    }
    #endif
    // Return the output value x FT_QUANT. We use FT_QUANT as the ONE value in later layers as well
    return (sum / FT_QUANT + bias) / L1_QUANT;
}

int32_t NNUE::flattenL2(const int32_t* inputs, const int16_t *weights, const int16_t bias) {
    int32_t sum = 0;
    #if defined(USE_AVX512)
    auto vec_sum = _mm512_setzero_si512();
    for (int i = 0; i < L2_SIZE / CHUNK_SIZE; i++) {
        auto weights_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights + i * CHUNK_SIZE));
        auto min = _mm512_set1_epi32(0);
        auto max = _mm512_set1_epi32(FT_QUANT);

        auto in_vec0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(inputs + (2 * i) * CHUNK_SIZE));
        auto in_vec1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(inputs + (2 * i + 1) * CHUNK_SIZE));
        auto clipped0 = _mm512_min_epi32(_mm512_max_epi32(in_vec0, min), max);
        auto clipped1 = _mm512_min_epi32(_mm512_max_epi32(in_vec1, min), max);
        auto clipped = _mm512_packs_epi32(clipped0, clipped1);

        auto mul = _mm512_madd_epi16(_mm512_mullo_epi16(clipped, weights_vec), clipped);
        vec_sum = _mm512_add_epi32(vec_sum, mul);
    }
    sum = _mm512_reduce_add_epi32(vec_sum);
    #elif defined(USE_AVX2)
    auto vec_sum = _mm256_setzero_si256();
    for (int i = 0; i < L2_SIZE / CHUNK_SIZE; i++) {
        auto weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i * CHUNK_SIZE));
        auto min = _mm256_set1_epi32(0);
        auto max = _mm256_set1_epi32(FT_QUANT);

        auto in_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + (2 * i) * CHUNK_SIZE));
        auto in_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + (2 * i + 1) * CHUNK_SIZE));
        auto clipped0 = _mm256_min_epi32(_mm256_max_epi32(in_vec0, min), max);
        auto clipped1 = _mm256_min_epi32(_mm256_max_epi32(in_vec1, min), max);
        auto clipped = _mm256_packs_epi32(clipped0, clipped1);

        auto mul = _mm256_madd_epi16(_mm256_mullo_epi16(clipped, weights_vec), clipped);
        vec_sum = _mm256_add_epi32(vec_sum, mul);
    }
    sum = horizontal_add(vec_sum);
    #else
    for (int i = 0; i < L2_SIZE; i++) {
        int32_t clipped = std::clamp(static_cast<int32_t>(inputs[i]), 0, FT_QUANT);
        sum += clipped * clipped * static_cast<int32_t>(weights[i]);
    }
    #endif
    return (sum / FT_QUANT + bias) * NET_SCALE / (FT_QUANT * L2_QUANT);
}

int32_t NNUE::output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {
    // this function takes the net output for the current accumulators and returns the eval of the position
    // according to the net
    const int16_t* us;
    const int16_t* them;
    if (whiteToMove) {
        us = board_accumulator[0].data();
        them = board_accumulator[1].data();
    } else {
        us = board_accumulator[1].data();
        them = board_accumulator[0].data();
    }

    int32_t L1Outputs[L2_SIZE];
    for (int i = 0; i < L2_SIZE; ++i)
        L1Outputs[i] = flattenL1(us,
                                 them,
                                 net.L1Weights[outputBucket] + L1_SIZE * i,
                                 net.L1Weights[outputBucket] + L1_SIZE * (L2_SIZE + i),
                                 net.L1Biases[outputBucket][i]);

    return flattenL2(L1Outputs, net.L2Weights[outputBucket], net.L2Biases[outputBucket]);
}

NNUEIndices NNUE::GetIndex(const int piece, const int square) {
    constexpr std::size_t COLOR_STRIDE = 64 * 6;
    constexpr std::size_t PIECE_STRIDE = 64;
    int piecetype = GetPieceType(piece);
    int color = Color[piece];
    std::size_t whiteIdx = color * COLOR_STRIDE + piecetype * PIECE_STRIDE + (square ^ 0b111'000);
    std::size_t blackIdx = (1 ^ color) * COLOR_STRIDE + piecetype * PIECE_STRIDE + square;
    return {whiteIdx, blackIdx};
}
