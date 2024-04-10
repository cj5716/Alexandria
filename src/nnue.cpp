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
UnquantisedNetwork unquantisedNet;

// Thanks to Disservin for having me look at his code and Luecx for the
// invaluable help and the immense patience

void NNUE::init(const char* file) {

    // open the nn file
    FILE* nn = fopen(file, "rb");

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
        net.FTWeights[i] = static_cast<int16_t>(unquantisedNet.FTWeights[i] * QUANT);

    // Quantise FT Biases
    for (int i = 0; i < L1_SIZE; ++i)
        net.FTBiases[i] = static_cast<int16_t>(unquantisedNet.FTBiases[i] * QUANT);

    // Transpose L1 and L2 weights and biases
    for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {

        // Quantise L1 Weights
        for (int i = 0; i < 2 * L1_SIZE; ++i)
            for (int j = 0; j < L2_SIZE; ++j)
                net.L1Weights[bucket][j * 2 * L1_SIZE + i] = static_cast<int16_t>(unquantisedNet.L1Weights[i][bucket][j] * QUANT);

        // Quantise L1 Biases
        for (int i = 0; i < L2_SIZE; ++i)
            net.L1Biases[bucket][i] = static_cast<int16_t>(unquantisedNet.L1Biases[bucket][i] * QUANT * QUANT);

        // Quantise L2 Weights
        for (int i = 0; i < L2_SIZE; ++i)
            net.L2Weights[bucket][i] = static_cast<int16_t>(unquantisedNet.L2Weights[i][bucket] * QUANT);

        // Quantise L2 Biases
        net.L2Biases[bucket] = static_cast<int16_t>(unquantisedNet.L2Biases[bucket] * QUANT * QUANT);
    }

}

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

int32_t NNUE::flattenL1(const int16_t *inputs, const int16_t *weights, const int16_t bias) {
    int32_t sum = 0;
    #if defined(USE_AVX512)
    __m512i vecSum = _mm512_setzero_si512();
    constexpr int32_t CHUNK_SIZE = sizeof(__m512i) / sizeof(int16_t);
    const __m512i zeroVec = _mm512_set1_epi16(0);
    const __m512i oneVec = _mm512_set1_epi16(QUANT);

    for (int i = 0; i < 2 * L1_SIZE / CHUNK_SIZE; i++) {
        __m512i weightsVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights + i * CHUNK_SIZE));
        __m512i inputsVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(inputs + i * CHUNK_SIZE));
        __m512i clippedVec = _mm512_min_epi16(_mm512_max_epi16(inputsVec, zeroVec), oneVec);
        __m512i productVec = _mm512_madd_epi16(_mm512_mullo_epi16(clippedVec, clippedVec), weightsVec);
        vecSum = _mm512_add_epi32(vecSum, productVec);
    }
    sum = _mm512_reduce_add_epi32(vecSum);

    #elif defined(USE_AVX2)
    __m256i vecSum = _mm256_setzero_si256();
    constexpr int32_t CHUNK_SIZE = sizeof(__m256i) / sizeof(int16_t);
    const __m256i zeroVec = _mm256_set1_epi16(0);
    const __m256i oneVec = _mm256_set1_epi16(QUANT);

    for (int i = 0; i < 2 * L1_SIZE / CHUNK_SIZE; i++) {
        __m256i weightsVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i * CHUNK_SIZE));
        __m256i inputsVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + i * CHUNK_SIZE));
        __m256i clippedVec = _mm256_min_epi16(_mm256_max_epi16(inputsVec, zeroVec), oneVec);
        __m256i productVec = _mm256_madd_epi16(_mm256_mullo_epi16(clippedVec, clippedVec), weightsVec);
        vecSum = _mm256_add_epi32(vecSum, productVec);
    }
    sum = horizontal_add(vecSum);
    #else
    for (int i = 0; i < 2 * L1_SIZE; i++) {
        int32_t clipped = std::clamp(static_cast<int32_t>(inputs[i]), 0, QUANT);
        sum += clipped * clipped * static_cast<int32_t>(weights[i]);
    }
    #endif
    // Return the output value x QUANT. We use QUANT as the ONE value in later layers as well
    return (sum / QUANT + bias) / QUANT;
}

int32_t NNUE::flattenL2(const int32_t* inputs, const int16_t *weights, const int16_t bias) {

    int32_t sum = 0;

    for (int i = 0; i < L2_SIZE; i++) {
        int32_t clipped = std::clamp(static_cast<int32_t>(inputs[i]), 0, QUANT);
        sum += clipped * clipped * static_cast<int32_t>(weights[i]);
    }

    return (sum / QUANT + bias) * NET_SCALE / (QUANT * QUANT);
}

int32_t NNUE::output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {

    const std::array<int16_t, L1_SIZE> ourAcc = board_accumulator[!whiteToMove];
    const std::array<int16_t, L1_SIZE> theirAcc = board_accumulator[whiteToMove];
    std::array<int16_t, L1_SIZE * 2> bothAccs = {};

    for (int i = 0; i < L1_SIZE; ++i)
    {
        bothAccs[i] = ourAcc[i];
        bothAccs[i + L1_SIZE] = theirAcc[i];
    }

    int32_t L1Outputs[L2_SIZE];
    for (int i = 0; i < L2_SIZE; ++i)
        L1Outputs[i] = flattenL1(bothAccs.data(),
                                 net.L1Weights[outputBucket] + i * 2 * L1_SIZE,
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
