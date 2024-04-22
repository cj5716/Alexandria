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

        read += fread(unquantisedNet.L1Weights, sizeof(float), OUTPUT_BUCKETS * 2 * L1_SIZE, nn);
        read += fread(unquantisedNet.L1Biases, sizeof(float), OUTPUT_BUCKETS, nn);

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

        std::memcpy(unquantisedNet.L1Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE;
        std::memcpy(unquantisedNet.L1Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS;
    }

    // Quantise FT Weights
    for (int i = 0; i < INPUT_SIZE * L1_SIZE; ++i)
        net.FTWeights[i] = static_cast<int16_t>(unquantisedNet.FTWeights[i] * FT_QUANT);

    // Quantise FT Biases
    for (int i = 0; i < L1_SIZE; ++i)
        net.FTBiases[i] = static_cast<int16_t>(unquantisedNet.FTBiases[i] * FT_QUANT);

    // Transpose L1 weights and biases
    for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {

        // Quantise L1 Weights
        for (int i = 0; i < 2 * L1_SIZE; ++i)
            net.L1Weights[bucket][i] = static_cast<int16_t>(unquantisedNet.L1Weights[i][bucket] * L1_QUANT);

        // Quantise L1 Biases
        net.L1Biases[bucket] = static_cast<int32_t>(unquantisedNet.L1Biases[bucket] * FT_QUANT * L1_QUANT);
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

#if defined(USE_AVX2)
int NNUE::hadd_int32(const __m256i sum) {
    __m128i upper_128 = _mm256_extracti128_si256(sum, 1);
    __m128i lower_128 = _mm256_castsi256_si128(sum);
    __m128i sum_128 = _mm_add_epi32(upper_128, lower_128);

    __m128i upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
    __m128i sum_64 = _mm_add_epi32(upper_64, sum_128);

    __m128i upper_32 = _mm_shuffle_epi32(sum_64, 1);
    __m128i sum_32 = _mm_add_epi32(upper_32, sum_64);

    return _mm_cvtsi128_si32(sum_32);
}
#endif

void NNUE::ActivateFTAndAffineL1(const int16_t *inputs, const int16_t *weights, const int32_t bias, int &output) {
    int sum = 0;
    #if defined(USE_AVX512)
    __m512i sumVec = _mm512_setzero_si512();
    const __m512i zeroVec = _mm512_set1_epi16(0);
    const __m512i oneVec = _mm512_set1_epi16(FT_QUANT);
    #elif defined(USE_AVX2)
    __m256i sumVec = _mm256_setzero_si256();
    const __m256i zeroVec = _mm256_set1_epi16(0);
    const __m256i oneVec = _mm256_set1_epi16(FT_QUANT);
    #else
    constexpr int ZERO = 0;
    constexpr int ONE = FT_QUANT;
    #endif
    for (int i = 0; i < 2 * L1_SIZE / L1_CHUNK_SIZE; ++i) {
        #if defined(USE_AVX512)
        const __m512i weightsVec = _mm512_loadu_si512(weights + i * L1_CHUNK_SIZE);
        const __m512i inputsVec = _mm512_loadu_si512(inputs + i * L1_CHUNK_SIZE);
        const __m512i clippedVec = _mm512_min_epi16(_mm512_max_epi16(inputsVec, zeroVec), oneVec);
        const __m512i squaredVec = _mm512_mullo_epi16(clippedVec, clippedVec);
        const __m512i productVec = _mm512_madd_epi16(squaredVec, weightsVec);
        sumVec = _mm512_add_epi32(sumVec, productVec);
        #elif defined(USE_AVX2)
        const __m256i weightsVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i * L1_CHUNK_SIZE));
        const __m256i inputsVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + i * L1_CHUNK_SIZE));
        const __m256i clippedVec = _mm256_min_epi16(_mm256_max_epi16(inputsVec, zeroVec), oneVec);
        const __m256i squaredVec = _mm256_mullo_epi16(clippedVec, clippedVec);
        const __m256i productVec = _mm256_madd_epi16(squaredVec, weightsVec);
        sumVec = _mm256_add_epi32(sumVec, productVec);
        #else
        const int clipped = std::clamp(static_cast<int>(inputs[i]), ZERO, ONE);
        const int squared = clipped * clipped;
        sum += squared * weights[i];
        #endif
    }
    #if defined(USE_AVX512)
    sum = _mm512_reduce_add_epi32(sumVec);
    #elif defined(USE_AVX2)
    sum = hadd_int32(sumVec);
    #endif
    output = (sum / FT_QUANT + bias) * NET_SCALE / (FT_QUANT * L1_QUANT);
}

int NNUE::output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {

    const std::array<int16_t, L1_SIZE> ourAcc = board_accumulator[!whiteToMove];
    const std::array<int16_t, L1_SIZE> theirAcc = board_accumulator[whiteToMove];
    std::array<int16_t, L1_SIZE * 2> bothAccs = {};

    for (int i = 0; i < L1_SIZE; ++i)
    {
        bothAccs[i] = ourAcc[i];
        bothAccs[i + L1_SIZE] = theirAcc[i];
    }

    int output;
    ActivateFTAndAffineL1(bothAccs.data(),
                          net.L1Weights[outputBucket],
                          net.L1Biases[outputBucket],
                          output);

    return output;
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
