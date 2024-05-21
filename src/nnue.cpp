#include "nnue.h"
#include "simd.h"
#include "position.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "incbin/incbin.h"

#if defined(USE_SIMD)
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

    UnquantisedNetwork *unquantisedNet = static_cast<UnquantisedNetwork*>(malloc(sizeof(UnquantisedNetwork)));

    // open the nn file
    FILE* nn = fopen(file, "rb");

    // if it's not invalid read the config values from it
    if (nn) {
        // initialize an accumulator for every input of the second layer
        size_t read = 0;
        const size_t fileSize = sizeof(UnquantisedNetwork);
        const size_t objectsExpected = fileSize / sizeof(float);

        read += fread(unquantisedNet->FTWeights, sizeof(float), INPUT_SIZE * L1_SIZE, nn);
        read += fread(unquantisedNet->FTBiases, sizeof(float), L1_SIZE, nn);

        read += fread(unquantisedNet->L1Weights, sizeof(float), OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE, nn);
        read += fread(unquantisedNet->L1Biases, sizeof(float), OUTPUT_BUCKETS * L2_SIZE, nn);

        read += fread(unquantisedNet->L2Weights, sizeof(float), OUTPUT_BUCKETS * L2_SIZE * L3_SIZE, nn);
        read += fread(unquantisedNet->L2Biases, sizeof(float), OUTPUT_BUCKETS * L3_SIZE, nn);

        read += fread(unquantisedNet->L3Weights, sizeof(float), OUTPUT_BUCKETS * L3_SIZE, nn);
        read += fread(unquantisedNet->L3Biases, sizeof(float), OUTPUT_BUCKETS, nn);

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
        std::memcpy(unquantisedNet->FTWeights, &gEVALData[memoryIndex], sizeof(float) * INPUT_SIZE * L1_SIZE);
        memoryIndex += sizeof(float) * INPUT_SIZE * L1_SIZE;
        std::memcpy(unquantisedNet->FTBiases, &gEVALData[memoryIndex], sizeof(float) * L1_SIZE);
        memoryIndex += sizeof(float) * L1_SIZE;

        std::memcpy(unquantisedNet->L1Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * 2 * L1_SIZE * L2_SIZE;
        std::memcpy(unquantisedNet->L1Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L2_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L2_SIZE;

        std::memcpy(unquantisedNet->L2Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L2_SIZE * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L2_SIZE * L3_SIZE;
        std::memcpy(unquantisedNet->L2Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L3_SIZE;

        std::memcpy(unquantisedNet->L3Weights, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS * L3_SIZE);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS * L3_SIZE;
        std::memcpy(unquantisedNet->L3Biases, &gEVALData[memoryIndex], sizeof(float) * OUTPUT_BUCKETS);
        memoryIndex += sizeof(float) * OUTPUT_BUCKETS;
    }

    // Quantise FT Weights
    for (int i = 0; i < INPUT_SIZE * L1_SIZE; ++i)
        net.FTWeights[i] = static_cast<int16_t>(std::round(unquantisedNet->FTWeights[i] * FT_QUANT));

    // Quantise FT Biases
    for (int i = 0; i < L1_SIZE; ++i)
        net.FTBiases[i] = static_cast<int16_t>(std::round(unquantisedNet->FTBiases[i] * FT_QUANT));

    // Transpose L1, L2 and L3 weights and biases
    for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {

        // Quantise L1 Weights
        #if defined(USE_SIMD)
        for (int i = 0; i < 2 * L1_SIZE / L1_CHUNK_SIZE; ++i)
            for (int j = 0; j < L2_SIZE; ++j)
                for (int k = 0; k < L1_CHUNK_SIZE; ++k)
                    net.L1Weights[bucket][  i * L1_CHUNK_SIZE * L2_SIZE
                                          + j * L1_CHUNK_SIZE
                                          + k] = static_cast<int16_t>(std::round(unquantisedNet->L1Weights[i * L1_CHUNK_SIZE + k][bucket][j] * L1_QUANT));
        #else
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < L2_SIZE; ++j)
                for (int k = 0; k < L1_SIZE; ++k)
                    net.L1Weights[bucket][  i * L1_SIZE * L2_SIZE
                                          + j * L1_SIZE
                                          + k] = static_cast<int16_t>(std::round(unquantisedNet->L1Weights[i * L1_SIZE + k][bucket][j] * L1_QUANT));
        #endif

        // Quantise L1 Biases
        for (int i = 0; i < L2_SIZE; ++i)
            net.L1Biases[bucket][i] = unquantisedNet->L1Biases[bucket][i];

        // Quantise L2 Weights
        #if defined(USE_SIMD)
        for (int i = 0; i < L2_SIZE / L2_CHUNK_SIZE; ++i)
            for (int j = 0; j < L3_SIZE; ++j)
                for (int k = 0; k < L2_CHUNK_SIZE; ++k)
                    net.L2Weights[bucket][  i * L2_CHUNK_SIZE * L3_SIZE
                                          + j * L2_CHUNK_SIZE
                                          + k] = unquantisedNet->L2Weights[i * L2_CHUNK_SIZE + k][bucket][j];
        #else
        for (int i = 0; i < L2_SIZE; ++i)
            for (int j = 0; j < L3_SIZE; ++j)
                net.L2Weights[bucket][j * L2_SIZE + i] = unquantisedNet->L2Weights[i][bucket][j];
        #endif

        // Quantise L2 Biases
        for (int i = 0; i < L3_SIZE; ++i)
            net.L2Biases[bucket][i] = unquantisedNet->L2Biases[bucket][i];

        // Quantise L3 Weights
        for (int i = 0; i < L3_SIZE; ++i)
            net.L3Weights[bucket][i] = unquantisedNet->L3Weights[i][bucket];

        // Quantise L3 Biases
        net.L3Biases[bucket] = unquantisedNet->L3Biases[bucket];
    }

    free(unquantisedNet);
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

void NNUE::update(Position *pos, std::vector<NNUEIndices>& NNUEAdd, std::vector<NNUEIndices>& NNUESub) {
    int adds = NNUEAdd.size();
    int subs = NNUESub.size();

    if (adds == 0 && subs == 0)
        return;

    // Quiets
    if (adds == 1 && subs == 1) {
        addSub(pos->accumStack[pos->accumStackHead - 1], pos->accumStack[pos->accumStackHead - 2], NNUEAdd[0], NNUESub[0]);
    }
    // Captures
    else if (adds == 1 && subs == 2) {
        addSubSub(pos->accumStack[pos->accumStackHead - 1], pos->accumStack[pos->accumStackHead - 2], NNUEAdd[0], NNUESub[0], NNUESub[1]);
    }
    // Castling
    else {
        addSub(pos->accumStack[pos->accumStackHead - 1], pos->accumStack[pos->accumStackHead - 2], NNUEAdd[0], NNUESub[0]);
        addSub(pos->accumStack[pos->accumStackHead - 1], pos->accumStack[pos->accumStackHead - 1], NNUEAdd[1], NNUESub[1]);
        // Note that for second addSub, we put accumStack[pos->accumStackHead - 1] instead of 2 because we are updating on top of
        // the half-updated accumulator
    }
    // Reset the add and sub vectors
    NNUEAdd.clear();
    NNUESub.clear();
}

void NNUE::addSub(NNUE::accumulator& new_acc, NNUE::accumulator& prev_acc, NNUEIndices add, NNUEIndices sub) {
    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx, blackSubIdx] = sub;
    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub = &net.FTWeights[whiteSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc[0][i] = prev_acc[0][i] - whiteSub[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub = &net.FTWeights[blackSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc[1][i] = prev_acc[1][i] - blackSub[i] + blackAdd[i];
    }
}

void NNUE::addSubSub(NNUE::accumulator& new_acc, NNUE::accumulator& prev_acc, NNUEIndices add, NNUEIndices sub1, NNUEIndices sub2) {

    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx1, blackSubIdx1] = sub1;
    auto [whiteSubIdx2, blackSubIdx2] = sub2;

    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub1 = &net.FTWeights[whiteSubIdx1 * L1_SIZE];
    auto whiteSub2 = &net.FTWeights[whiteSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc[0][i] = prev_acc[0][i] - whiteSub1[i] - whiteSub2[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub1 = &net.FTWeights[blackSubIdx1 * L1_SIZE];
    auto blackSub2 = &net.FTWeights[blackSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc[1][i] = prev_acc[1][i] - blackSub1[i] - blackSub2[i] + blackAdd[i];
    }
}

void NNUE::ActivateFTAndPropagateL1(const int16_t *us, const int16_t *them, const int16_t *weights, const float *biases, float *output) {
    #if defined(USE_SIMD)
    VecEpi sumVecs[L2_SIZE] = {};
    const VecEpi Zero = vec_set1_epi16(0);
    const VecEpi One  = vec_set1_epi16(FT_QUANT);
    int weightIdx = 0;

    for (const int16_t *acc : {us, them}) {
        for (int i = 0; i < L1_SIZE; i += L1_CHUNK_SIZE) {
            const VecEpi *weightVecs = reinterpret_cast<const VecEpi*>(&weights[weightIdx + i * L2_SIZE]);
            const VecEpi inputVec    = vec_loadu_epi(reinterpret_cast<const VecEpi*>(&acc[i]));
            const VecEpi clippedVec  = vec_max_epi16(vec_min_epi16(inputVec, One), Zero);

            // What we want to achieve is to get the product of the 2 vectors, shifted right by 3,
            // without relying on intermediate epi32 vectors. To do this, we shift left by (6+7=13)
            // before the multiplication, before using mulhi (which strips out the bottom 16 bits,
            // effectively shifting right by 16). This process effectively results in a net shift
            // of 3 to the right without losing precision by shifting right before the multiplication,
            // or giving a large slowdown due to intermediate epi32 unpacking and repacking.
            const VecEpi clippedVec0 = vec_slli_epi16(clippedVec, 6);
            const VecEpi clippedVec1 = vec_slli_epi16(clippedVec, 7);
            const VecEpi squaredVec  = vec_mulhi_epu16(clippedVec0, clippedVec1);

            for (int out = 0; out < L2_SIZE; ++out) {
                const VecEpi productVec = vec_madd_epi16(squaredVec, weightVecs[out]);
                sumVecs[out] = vec_add_epi32(sumVecs[out], productVec);
            }
        }
        weightIdx += L1_SIZE * L2_SIZE;
    }

    for (int i = 0; i < L2_SIZE; i += L2_CHUNK_SIZE) {
        const Vec256Epi sum0123 = vec256_hadd_epi32x4(reinterpret_cast<const VecEpi*>(&sumVecs[i]));
        const Vec256Epi sum4567 = vec256_hadd_epi32x4(reinterpret_cast<const VecEpi*>(&sumVecs[i + 4]));
        const Vec256Epi sum     = vec256_comb_epi32(sum0123, sum4567);
        const VecPs     biasVec = vec_loadu_ps(&biases[i]);
        const VecPs     sumDiv  = vec_set1_ps(float(FT_QUANT * FT_QUANT * L1_QUANT) / 8.0f);
        const VecPs     sumPs   = vec_add_ps(vec_div_ps(vec_cvtepi_ps(sum), sumDiv), biasVec);
        const VecPs     L1Zero  = vec_set1_ps(0.0f);
        const VecPs     L1One   = vec_set1_ps(1.0f);
        const VecPs     clipped = vec_min_ps(vec_max_ps(sumPs, L1Zero), L1One);
        const VecPs     squared = vec_mul_ps(clipped, clipped);
        vec_storeu_ps(&output[i], squared);
    }
    #else
    int sums[L2_SIZE] = {};
    for (int i = 0; i < L1_SIZE; ++i) {
        const int16_t clippedUs   = std::clamp(us[i], int16_t(0), FT_QUANT);
        const int16_t clippedThem = std::clamp(them[i], int16_t(0), FT_QUANT);
        const int16_t squaredUs   = (clippedUs * clippedUs) >> 3;
        const int16_t squaredThem = (clippedThem * clippedThem) >> 3;
        for (int out = 0; out < L2_SIZE; ++out) {
            sums[out] += squaredUs   * weights[out * L1_SIZE + i];
            sums[out] += squaredThem * weights[out * L1_SIZE + i + L1_SIZE * L2_SIZE];
        }
    }

    for (int i = 0; i < L2_SIZE; ++i) {
        const float sumDiv  = float(FT_QUANT * FT_QUANT * L1_QUANT) / 8.0f;
        const float clipped = std::clamp(float(sums[i]) / sumDiv + biases[i], 0.0f, 1.0f);
        output[i] = clipped * clipped;
    }
    #endif
}

void NNUE::PropagateL2(const float *inputs, const float *weights, const float *biases, float *output) {
    #if defined(USE_SIMD)
    VecPs sumVecs[L3_SIZE] = {};
    for (int i = 0; i < L2_SIZE; i += L2_CHUNK_SIZE) {
        const VecPs *weightVecs = reinterpret_cast<const VecPs*>(&weights[i * L3_SIZE]);
        const VecPs inputsVec   = vec_loadu_ps(&inputs[i]);
        for (int out = 0; out < L3_SIZE; ++out)
            sumVecs[out] = vec_mul_add_ps(inputsVec, weightVecs[out], sumVecs[out]);
    }

    for (int i = 0; i < L3_SIZE; i += L3_CHUNK_SIZE) {
        const VecPs sum0123 = vec_hadd_psx4(&sumVecs[i]);
        const VecPs sum4567 = vec_hadd_psx4(&sumVecs[i + 4]);
        const VecPs biasVec = vec_loadu_ps(&biases[i]);
        const VecPs sum     = vec_add_ps(vec_comb_ps(sum0123, sum4567), biasVec);
        const VecPs Zero    = vec_set1_ps(0.0f);
        const VecPs One     = vec_set1_ps(1.0f);
        const VecPs clipped = vec_max_ps(vec_min_ps(sum, One), Zero);
        const VecPs squared = vec_mul_ps(clipped, clipped);
        vec_storeu_ps(&output[i], squared);
    }
    #else
    float sums[L3_SIZE];

    for (int i = 0; i < L3_SIZE; ++i)
        sums[i] = biases[i];

    for (int i = 0; i < L2_SIZE; ++i) {
        for (int out = 0; out < L3_SIZE; ++out) {
            sums[out] += inputs[i] * weights[out * L2_SIZE + i];
        }
    }

    for (int i = 0; i < L3_SIZE; ++i) {
        const float clipped = std::clamp(sums[i], 0.0f, 1.0f);
        output[i] = clipped * clipped;
    }
    #endif
}

void NNUE::PropagateL3(const float *inputs, const float *weights, const float bias, float &output) {
    #if defined(USE_SIMD)
    VecPs sumVec = vec_set1_ps(0.0f);
    for (int i = 0; i < L3_SIZE; i += L3_CHUNK_SIZE) {
        const VecPs weightVec = vec_loadu_ps(&weights[i]);
        const VecPs inputsVec = vec_loadu_ps(&inputs[i]);
        sumVec = vec_mul_add_ps(inputsVec, weightVec, sumVec);
    }
    output = bias + vec_reduce_add_ps(sumVec);
    #else
    float sum = bias;
    for (int i = 0; i < L3_SIZE; ++i) {
        sum += inputs[i] * weights[i];
    }
    output = sum;
    #endif
}

int NNUE::output(const NNUE::accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {

    float L1Outputs[L2_SIZE];
    float L2Outputs[L3_SIZE];
    float L3Output;

    ActivateFTAndPropagateL1(board_accumulator[!whiteToMove].data(),
                             board_accumulator[whiteToMove].data(),
                             net.L1Weights[outputBucket],
                             net.L1Biases[outputBucket],
                             L1Outputs);

    PropagateL2(L1Outputs,
                net.L2Weights[outputBucket],
                net.L2Biases[outputBucket],
                L2Outputs);

    PropagateL3(L2Outputs,
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