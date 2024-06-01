#include "nnue.h"
#include "simd.h"
#include <algorithm>
#include "position.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include "incbin/incbin.h"

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

    // if it's not invalid read the config values from it
    if (nn) {
        // initialize an accumulator for every input of the second layer
        size_t read = 0;
        const size_t fileSize = sizeof(Network) - PADDING_SIZE;
        const size_t objectsExpected = fileSize / sizeof(int16_t);

        read += fread(net.FTWeights, sizeof(int16_t), NUM_INPUTS * L1_SIZE, nn);
        read += fread(net.FTBiases, sizeof(int16_t), L1_SIZE, nn);
        read += fread(net.L1Weights, sizeof(int16_t), L1_SIZE * 2 * OUTPUT_BUCKETS, nn);
        read += fread(net.L1Biases, sizeof(int16_t), OUTPUT_BUCKETS, nn);

        if (read != objectsExpected) {
            std::cout << "Error loading the net, aborting ";
            std::cout << "Expected " << objectsExpected << " shorts, got " << read << "\n";
            exit(1);
        }

        // after reading the config we can close the file
        fclose(nn);
    } else {
        // if we don't find the nnue file we use the net embedded in the exe
        uint64_t memoryIndex = 0;
        std::memcpy(net.FTWeights, &gEVALData[memoryIndex], NUM_INPUTS * L1_SIZE * sizeof(int16_t));
        memoryIndex += NUM_INPUTS * L1_SIZE * sizeof(int16_t);
        std::memcpy(net.FTBiases, &gEVALData[memoryIndex], L1_SIZE * sizeof(int16_t));
        memoryIndex += L1_SIZE * sizeof(int16_t);

        std::memcpy(net.L1Weights, &gEVALData[memoryIndex], L1_SIZE * 2 * OUTPUT_BUCKETS * sizeof(int16_t));
        memoryIndex += L1_SIZE * 2 * OUTPUT_BUCKETS * sizeof(int16_t);
        std::memcpy(net.L1Biases, &gEVALData[memoryIndex], OUTPUT_BUCKETS * sizeof(int16_t));
    }

    int16_t transposedL1Weights[L1_SIZE * 2 * OUTPUT_BUCKETS];
    for (int weight = 0; weight < 2 * L1_SIZE; ++weight)
    {
        for (int bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket)
        {
            const int srcIdx = weight * OUTPUT_BUCKETS + bucket;
            const int dstIdx = bucket * 2 * L1_SIZE + weight;
            transposedL1Weights[dstIdx] = net.L1Weights[srcIdx];
        }
    }
    std::memcpy(net.L1Weights, transposedL1Weights, L1_SIZE * sizeof(int16_t) * 2 * OUTPUT_BUCKETS);

}

void NNUE::accumulate(NNUE::Accumulator& board_accumulator, Position* pos) {
    for (int i = 0; i < L1_SIZE; i++) {
        board_accumulator.values[0][i] = net.FTBiases[i];
        board_accumulator.values[1][i] = net.FTBiases[i];
    }

    for (int i = 0; i < 64; i++) {
        bool input = pos->pieces[i] != EMPTY;
        if (!input) continue;
        auto [whiteIdx, blackIdx] = GetIndex(pos->pieces[i], i);
        auto whiteAdd = &net.FTWeights[whiteIdx * L1_SIZE];
        auto blackAdd = &net.FTWeights[blackIdx * L1_SIZE];
        for (int j = 0; j < L1_SIZE; j++) {
            board_accumulator.values[0][j] += whiteAdd[j];
        }
        for (int j = 0; j < L1_SIZE; j++) {
            board_accumulator.values[1][j] += blackAdd[j];
        }
    }
}

void NNUE::update(NNUE::Accumulator *acc) {

    int adds = acc->NNUEAdd.size();
    int subs = acc->NNUESub.size();

    if (adds == 0 && subs == 0)
        return;

    if (!(acc - 1)->NNUEAdd.empty() && !(acc - 1)->NNUESub.empty())
        update(acc - 1);

    // Quiets
    if (adds == 1 && subs == 1) {
        addSub(acc, acc - 1, acc->NNUEAdd[0], acc->NNUESub[0]);
    }
    // Captures
    else if (adds == 1 && subs == 2) {
        addSubSub(acc, acc - 1, acc->NNUEAdd[0], acc->NNUESub[0], acc->NNUESub[1]);
    }
    // Castling
    else {
        addSub(acc, acc - 1, acc->NNUEAdd[0], acc->NNUESub[0]);
        addSub(acc, acc, acc->NNUEAdd[1], acc->NNUESub[1]);
        // Note that for second addSub, we put acc instead of acc - 1 because we are updating on top of
        // the half-updated accumulator
    }
    // Reset the add and sub vectors
    acc->NNUEAdd.clear();
    acc->NNUESub.clear();
}

void NNUE::addSub(NNUE::Accumulator *new_acc, NNUE::Accumulator *prev_acc, NNUEIndices add, NNUEIndices sub) {
    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx, blackSubIdx] = sub;

    int16_t *adds[2] = {&net.FTWeights[whiteAddIdx * L1_SIZE], &net.FTWeights[blackAddIdx * L1_SIZE]};
    int16_t *subs[2] = {&net.FTWeights[whiteSubIdx * L1_SIZE], &net.FTWeights[blackSubIdx * L1_SIZE]};

    #if defined(USE_SIMD)
    vepi16 accRegs[TILE_REGS];
    for (int side = WHITE; side < BOTH; ++side) {
        for (int i = 0; i < L1_SIZE; i += TILE_REGS * CHUNK_SIZE) {
            const vepi16 *prevAccTile = reinterpret_cast<const vepi16*>(&prev_acc->values[side][i]);
            const vepi16 *addTile     = reinterpret_cast<const vepi16*>(&adds[side][i]);
            const vepi16 *subTile     = reinterpret_cast<const vepi16*>(&subs[side][i]);

            for (int j = 0; j < TILE_REGS; ++j)
                accRegs[j] = vec_load_epi(&prevAccTile[j]);

            for (int j = 0; j < TILE_REGS; ++j)
                accRegs[j] = vec_add_epi16(accRegs[j], vec_sub_epi16(addTile[j], subTile[j]));

            vepi16 *newAccTile  = reinterpret_cast<vepi16*>(&new_acc->values[side][i]);
            for (int j = 0; j < TILE_REGS; ++j)
                vec_store_epi(&newAccTile[j], accRegs[j]);
        }
    }
    #else
    for (int side = WHITE; side < BOTH; ++side) {
        for (int i = 0; i < L1_SIZE; ++i) {
            new_acc->values[side][i] = prev_acc->values[side][i] + adds[side][i] - subs[side][i];
        }
    }
    #endif
}

void NNUE::addSubSub(NNUE::Accumulator *new_acc, NNUE::Accumulator *prev_acc, NNUEIndices add, NNUEIndices sub1, NNUEIndices sub2) {

    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx1, blackSubIdx1] = sub1;
    auto [whiteSubIdx2, blackSubIdx2] = sub2;

    int16_t *adds [2] = {&net.FTWeights[whiteAddIdx  * L1_SIZE], &net.FTWeights[blackAddIdx  * L1_SIZE]};
    int16_t *subs1[2] = {&net.FTWeights[whiteSubIdx1 * L1_SIZE], &net.FTWeights[blackSubIdx1 * L1_SIZE]};
    int16_t *subs2[2] = {&net.FTWeights[whiteSubIdx2 * L1_SIZE], &net.FTWeights[blackSubIdx2 * L1_SIZE]};

    #if defined(USE_SIMD)
    vepi16 accRegs[TILE_REGS];
    for (int side = WHITE; side < BOTH; ++side) {
        for (int i = 0; i < L1_SIZE; i += TILE_REGS * CHUNK_SIZE) {
            const vepi16 *prevAccTile = reinterpret_cast<const vepi16*>(&prev_acc->values[side][i]);
            const vepi16 *addTile     = reinterpret_cast<const vepi16*>(&adds[side][i]);
            const vepi16 *subTile1    = reinterpret_cast<const vepi16*>(&subs1[side][i]);
            const vepi16 *subTile2    = reinterpret_cast<const vepi16*>(&subs2[side][i]);

            for (int j = 0; j < TILE_REGS; ++j)
                accRegs[j] = vec_load_epi(&prevAccTile[j]);

            for (int j = 0; j < TILE_REGS; ++j)
                accRegs[j] = vec_add_epi16(accRegs[j], vec_sub_epi16(vec_sub_epi16(addTile[j], subTile1[j]), subTile2[j]));

            vepi16 *newAccTile  = reinterpret_cast<vepi16*>(&new_acc->values[side][i]);
            for (int j = 0; j < TILE_REGS; ++j)
                vec_store_epi(&newAccTile[j], accRegs[j]);
        }
    }
    #else
    for (int side = WHITE; side < BOTH; ++side) {
        for (int i = 0; i < L1_SIZE; ++i) {
            new_acc->values[side][i] = prev_acc->values[side][i] + adds[side][i] - subs1[side][i] - subs2[side][i];
        }
    }
    #endif
}

int32_t NNUE::ActivateFTAndAffineL1(const int16_t *us, const int16_t *them, const int16_t *weights, const int16_t bias) {
    #if defined(USE_SIMD)
    vepi32 sum  = vec_zero_epi32();
    const vepi16 Zero = vec_zero_epi16();
    const vepi16 One  = vec_set1_epi16(FT_QUANT);
    int weightOffset = 0;
    for (const int16_t *acc : {us, them}) {
        for (int i = 0; i < L1_SIZE; i += CHUNK_SIZE) {
            vepi16 input   = vec_load_epi(reinterpret_cast<const vepi16*>(&acc[i]));
            vepi16 weight  = vec_load_epi(reinterpret_cast<const vepi16*>(&weights[i + weightOffset]));
            vepi16 clipped = vec_min_epi16(vec_max_epi16(input, Zero), One);

            // In squared clipped relu, we want to do (clipped * clipped) * weight.
            // However, as clipped * clipped does not fit in an int16 while clipped * weight does,
            // we instead do mullo(clipped, weight) and then madd by clipped.
            vepi32 product = vec_madd_epi16(vec_mullo_epi16(clipped, weight), clipped);
            sum = vec_add_epi32(sum, product);
        }

        weightOffset += L1_SIZE;
    }

    return (vec_reduce_add_epi32(sum) / FT_QUANT + bias) * NET_SCALE / (FT_QUANT * L1_QUANT);

    #else
    int sum = 0;
    int weightOffset = 0;
    for (const int16_t *acc : {us, them}) {
        for (int i = 0; i < L1_SIZE; ++i) {
            int16_t input   = acc[i];
            int16_t weight  = weights[i + weightOffset];
            int16_t clipped = std::clamp(input, int16_t(0), int16_t(FT_QUANT));
            sum += static_cast<int16_t>(clipped * weight) * clipped;
        }

        weightOffset += L1_SIZE;
    }

    return (sum / FT_QUANT + bias) * NET_SCALE / (FT_QUANT * L1_QUANT);
    #endif
}

int32_t NNUE::output(const NNUE::Accumulator& board_accumulator, const bool whiteToMove, const int outputBucket) {
    // this function takes the net output for the current accumulators and returns the eval of the position
    // according to the net
    const int16_t* us;
    const int16_t* them;
    if (whiteToMove) {
        us = board_accumulator.values[0].data();
        them = board_accumulator.values[1].data();
    } else {
        us = board_accumulator.values[1].data();
        them = board_accumulator.values[0].data();
    }

    const int32_t bucketOffset = 2 * L1_SIZE * outputBucket;
    return ActivateFTAndAffineL1(us, them, &net.L1Weights[bucketOffset], net.L1Biases[outputBucket]);
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
