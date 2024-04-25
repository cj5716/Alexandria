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

        read += fread(unquantisedNet.L2Weights, sizeof(float), OUTPUT_BUCKETS * L2_SIZE * L3_SIZE, nn);
        read += fread(unquantisedNet.L2Biases, sizeof(float), OUTPUT_BUCKETS * L3_SIZE, nn);

        read += fread(unquantisedNet.L3Weights, sizeof(float), OUTPUT_BUCKETS * L3_SIZE, nn);
        read += fread(unquantisedNet.L3Biases, sizeof(float), OUTPUT_BUCKETS, nn);

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

        std::memcpy(unquantisedNet.L2Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L2_SIZE * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L2_SIZE * L3_SIZE;
        std::memcpy(unquantisedNet.L2Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L3_SIZE;

        std::memcpy(unquantisedNet.L3Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L3_SIZE;
        std::memcpy(unquantisedNet.L3Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS;
    }

    // Quantise FT Weights
    for (int i = 0; i < INPUT_SIZE * L1_SIZE; ++i)
        net.FTWeights[i] = static_cast<int16_t>(unquantisedNet.FTWeights[i] * FT_QUANT);

    // Quantise FT Biases
    for (int i = 0; i < L1_SIZE; ++i)
        net.FTBiases[i] = static_cast<int16_t>(unquantisedNet.FTBiases[i] * FT_QUANT);

    // Transpose L1, L2 and L3 weights and biases
    for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {

        // Quantise L1 Weights
        for (int i = 0; i < 2 * L1_SIZE / L1_CHUNK_SIZE; ++i)
            for (int j = 0; j < L2_SIZE; ++j)
                for (int k = 0; k < L1_CHUNK_SIZE; ++k)
                    net.L1Weights[bucket][  i * L1_CHUNK_SIZE * L2_SIZE
                                          + j * L1_CHUNK_SIZE
                                          + k] = static_cast<int16_t>(unquantisedNet.L1Weights[i * L1_CHUNK_SIZE + k][bucket][j] * L1_QUANT);

        // Quantise L1 Biases
        for (int i = 0; i < L2_SIZE; ++i)
            net.L1Biases[bucket][i] = unquantisedNet.L1Biases[bucket][i];

        // Quantise L2 Weights
        for (int i = 0; i < L2_SIZE / L2_CHUNK_SIZE; ++i)
            for (int j = 0; j < L3_SIZE; ++j)
                for (int k = 0; k < L2_CHUNK_SIZE; ++k)
                    net.L2Weights[bucket][  i * L2_CHUNK_SIZE * L3_SIZE
                                          + j * L2_CHUNK_SIZE
                                          + k] = unquantisedNet.L2Weights[i * L2_CHUNK_SIZE + k][bucket][j];

        // Quantise L2 Biases
        for (int i = 0; i < L3_SIZE; ++i)
            net.L2Biases[bucket][i] = unquantisedNet.L2Biases[bucket][i];

        // Quantise L3 Weights
        for (int i = 0; i < L3_SIZE; ++i)
            net.L3Weights[bucket][i] = unquantisedNet.L3Weights[i][bucket];

        // Quantise L3 Biases
        net.L3Biases[bucket] = unquantisedNet.L3Biases[bucket];
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

#if defined(USE_AVX512) || defined(USE_AVX2)
float NNUE::_mm256_reduce_add_ps(const __m256 sum) {
    
    const __m128 upper_128 = _mm256_extractf128_ps(sum, 1);
    const __m128 lower_128 = _mm256_castps256_ps128(sum);
    const __m128 sum_128 = _mm_add_ps(upper_128, lower_128);

    const __m128 upper_64 = _mm_movehl_ps(sum_128, sum_128);
    const __m128 sum_64 = _mm_add_ps(upper_64, sum_128);

    const __m128 upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
    const __m128 sum_32 = _mm_add_ss(upper_32, sum_64);

    return _mm_cvtss_f32(sum_32);
}

__m256i NNUE::combine_m256i(const __m256i in0, const __m256i in1) {
    const __m128i in0_low = _mm256_castsi256_si128(in0);
    const __m128i in0_hi = _mm256_extracti128_si256(in0, 1);
    const __m128i in0_m128 = _mm_add_epi32(in0_low, in0_hi);

    const __m128i in1_low = _mm256_castsi256_si128(in1);
    const __m128i in1_hi = _mm256_extracti128_si256(in1, 1);
    const __m128i in1_m128 = _mm_add_epi32(in1_low, in1_hi);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(in0_m128), in1_m128, 1);
}

__m256 NNUE::combine_m256(const __m256 in0, const __m256 in1) {
    const __m128 in0_low = _mm256_castps256_ps128(in0);
    const __m128 in0_hi = _mm256_extractf128_ps(in0, 1);
    const __m128 in0_m128 = _mm_add_ps(in0_low, in0_hi);

    const __m128 in1_low = _mm256_castps256_ps128(in1);
    const __m128 in1_hi = _mm256_extractf128_ps(in1, 1);
    const __m128 in1_m128 = _mm_add_ps(in1_low, in1_hi);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(in0_m128), in1_m128, 1);
}

__m256 NNUE::hadd_psx4(const __m256* in) {
    const __m256 sum01 = _mm256_hadd_ps(in[0], in[1]);
    const __m256 sum23 = _mm256_hadd_ps(in[2], in[3]);
    return _mm256_hadd_ps(sum01, sum23);
}
#endif

#if defined(USE_AVX512)
__m256i NNUE::m512_to_m256(const __m512i in) {
    const __m256i upper256 = _mm512_extracti32x8_epi32(in, 1);
    const __m256i lower256 = _mm512_castsi512_si256(in);
    return _mm256_add_epi32(lower256, upper256);
}

__m256i NNUE::hadd_epi32x4(const __m512i* in) {
    const __m256i sum01 = _mm256_hadd_epi32(m512_to_m256(in[0]), m512_to_m256(in[1]));
    const __m256i sum23 = _mm256_hadd_epi32(m512_to_m256(in[2]), m512_to_m256(in[3]));
    return _mm256_hadd_epi32(sum01, sum23);
}
#elif defined(USE_AVX2)
__m256i NNUE::hadd_epi32x4(const __m256i* in) {
    const __m256i sum01 = _mm256_hadd_epi32(in[0], in[1]);
    const __m256i sum23 = _mm256_hadd_epi32(in[2], in[3]);
    return _mm256_hadd_epi32(sum01, sum23);
}
#endif

void NNUE::ActivateFTAndAffineL1(const int16_t *inputs, const int16_t *weights, int *output) {
    #if defined(USE_AVX512)
    __m512i sumVecs[L2_SIZE] = {};
    const __m512i *weightsVecs = reinterpret_cast<const __m512i*>(weights);
    const __m512i *inputsVecs = reinterpret_cast<const __m512i*>(inputs);
    const __m512i zeroVec = _mm512_set1_epi16(0);
    const __m512i oneVec = _mm512_set1_epi16(FT_QUANT);
    #elif defined(USE_AVX2)
    const __m256i *weightsVecs = reinterpret_cast<const __m256i*>(weights);
    const __m256i *inputsVecs = reinterpret_cast<const __m256i*>(inputs);
    __m256i sumVecs[L2_SIZE] = {};
    const __m256i zeroVec = _mm256_set1_epi16(0);
    const __m256i oneVec = _mm256_set1_epi16(FT_QUANT);
    #else
    int sums[L2_SIZE] = {};
    constexpr int ZERO = 0;
    constexpr int ONE = FT_QUANT;
    #endif
    for (int i = 0; i < L1_SIZE / L1_CHUNK_SIZE; ++i) {
        #if defined(USE_AVX512)
        const __m512i *weightsVec = &weightsVecs[i * L2_SIZE];
        const __m512i clippedVec = _mm512_min_epi16(_mm512_max_epi16(inputsVecs[i], zeroVec), oneVec);
        const __m512i squaredVec = _mm512_mullo_epi16(clippedVec, clippedVec);
        for (int out = 0; out < L2_SIZE; ++out) {
            const __m512i productVec = _mm512_madd_epi16(squaredVec, weightsVec[out]);
            sumVecs[out] = _mm512_add_epi32(sumVecs[out], productVec);
        }
        #elif defined(USE_AVX2)
        const __m256i *weightsVec = &weightsVecs[i * L2_SIZE];
        const __m256i clippedVec = _mm256_min_epi16(_mm256_max_epi16(inputsVecs[i], zeroVec), oneVec);
        const __m256i squaredVec = _mm256_mullo_epi16(clippedVec, clippedVec);
        for (int out = 0; out < L2_SIZE; ++out) {
            const __m256i productVec = _mm256_madd_epi16(squaredVec, weightsVec[out]);
            sumVecs[out] = _mm256_add_epi32(sumVecs[out], productVec);
        }
        #else
        const int clipped = std::clamp(static_cast<int>(inputs[i]), ZERO, ONE);
        const int squared = clipped * clipped;
        for (int out = 0; out < L2_SIZE; ++out)
            sums[out] += squared * weights[i * L2_SIZE + out];
        #endif
    }
    for (int i = 0; i < L2_SIZE / L2_CHUNK_SIZE; ++i) {
        #if defined(USE_AVX512) || defined(USE_AVX2)
        const __m256i sum0123 = hadd_epi32x4(&sumVecs[i * L2_CHUNK_SIZE]);
        const __m256i sum4567 = hadd_epi32x4(&sumVecs[i * L2_CHUNK_SIZE + 4]);
        const __m256i sum = combine_m256i(sum0123, sum4567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[i]), sum);
        #else
        output[i] = sums[i];
        #endif
    }
}

void NNUE::ActivateL1AndAffineL2(const float *inputs, const float *weights, const float *biases, float *output) {

    #if defined(USE_AVX512) || defined(USE_AVX2)
    __m256 sumVecs[L3_SIZE] = {};
    const __m256 zeroVec = _mm256_set1_ps(0.0f);
    const __m256 oneVec  = _mm256_set1_ps(1.0f);
    #else
    float sums[L3_SIZE] = {};
    constexpr float ZERO = 0.0f;
    constexpr float ONE  = 1.0f;
    #endif

    for (int i = 0; i < L2_SIZE / L2_CHUNK_SIZE; ++i) {
        #if defined(USE_AVX512) || defined(USE_AVX2)
        const __m256 *weightsVecs = reinterpret_cast<const __m256*>(weights + i * L3_SIZE * L2_CHUNK_SIZE);
        const __m256 inputsVec = _mm256_loadu_ps(inputs + i * L2_CHUNK_SIZE);
        const __m256 clippedVec = _mm256_min_ps(_mm256_max_ps(inputsVec, zeroVec), oneVec);
        const __m256 squaredVec = _mm256_mul_ps(clippedVec, clippedVec);
        for (int out = 0; out < L3_SIZE; ++out)
            sumVecs[out] = _mm256_fmadd_ps(squaredVec, weightsVecs[out], sumVecs[out]);
        #else
        const float clipped = std::clamp(inputs[i], ZERO, ONE);
        const float squared = clipped * clipped;
        for (int out = 0; out < L3_SIZE; ++out)
            sums[out] += squared * weights[i * L3_SIZE + out];
        #endif
    }

    for (int i = 0; i < L3_SIZE / L3_CHUNK_SIZE; ++i) {
        #if defined(USE_AVX512) || defined(USE_AVX2)
        const __m256 sum0123 = hadd_psx4(&sumVecs[i * L3_CHUNK_SIZE]);
        const __m256 sum4567 = hadd_psx4(&sumVecs[i * L3_CHUNK_SIZE + 4]);
        const __m256 bias = _mm256_loadu_ps(&biases[i * L3_CHUNK_SIZE]);
        const __m256 sum = _mm256_add_ps(combine_m256(sum0123, sum4567), bias);
        _mm256_storeu_ps(&output[i * L3_CHUNK_SIZE], sum);
        #else
        output[i] = sums[i] + biases[i];
        #endif
    }
}

void NNUE::ActivateL2AndAffineL3(const float *inputs, const float *weights, const float bias, float &output) {
    float sum = 0.0f;
    #if defined(USE_AVX512) || defined(USE_AVX2)
    __m256 sumVec = _mm256_setzero_ps();
    const __m256 zeroVec = _mm256_set1_ps(0.0f);
    const __m256 oneVec  = _mm256_set1_ps(1.0f);

    for (int i = 0; i < L3_SIZE / L3_CHUNK_SIZE; ++i) {
        const __m256 weightsVec = _mm256_loadu_ps(weights + i * L3_CHUNK_SIZE);
        const __m256 inputsVec = _mm256_loadu_ps(inputs + i * L3_CHUNK_SIZE);
        const __m256 clippedVec = _mm256_min_ps(_mm256_max_ps(inputsVec, zeroVec), oneVec);
        const __m256 squaredVec = _mm256_mul_ps(clippedVec, clippedVec);
        sumVec = _mm256_fmadd_ps(squaredVec, weightsVec, sumVec);
    }
    sum = _mm256_reduce_add_ps(sumVec);
    #else
    for (int i = 0; i < L3_SIZE; ++i) {
        const float clipped = std::clamp(inputs[i], 0.0f, 1.0f);
        sum += clipped * clipped * weights[i];
    }
    #endif
    output = sum + bias;
}

int NNUE::output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {

    int   L1OutputsUs[L2_SIZE];
    int   L1OutputsThem[L2_SIZE];
    float L2Inputs[L2_SIZE];
    float L2Outputs[L3_SIZE];
    float L3Output;

    ActivateFTAndAffineL1(board_accumulator[!whiteToMove].data(),
                          net.L1Weights[outputBucket],
                          L1OutputsUs);

    ActivateFTAndAffineL1(board_accumulator[whiteToMove].data(),
                          net.L1Weights[outputBucket] + L1_SIZE * L2_SIZE,
                          L1OutputsThem);

    for (int i = 0; i < L2_SIZE; ++i) {
        L2Inputs[i] = float(L1OutputsUs[i] + L1OutputsThem[i]) / float(FT_QUANT * FT_QUANT * L1_QUANT) + net.L1Biases[outputBucket][i];
    }

    ActivateL1AndAffineL2(L2Inputs,
                          net.L2Weights[outputBucket],
                          net.L2Biases[outputBucket],
                          L2Outputs);

    ActivateL2AndAffineL3(L2Outputs,
                          net.L3Weights[outputBucket],
                          net.L3Biases[outputBucket],
                          L3Output);

    return L3Output * NET_SCALE;
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
