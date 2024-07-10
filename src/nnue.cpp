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
        const size_t fileSize = sizeof(Network);
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

void NNUE::accumulate(NNUE::Accumulator &board_accumulator, Position* pos) {
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

    // The last updated accumulator is indicated by acc - i
    int i = 0;
    NNUE::Accumulator tmp_acc;

    // Find the last updated accumulator. It is updated if it does not have anything to add or sub
    while (true) {
        if (acc[-i].NNUEAdd.empty() && acc[-i].NNUESub.empty()) break;
        i++;
    }

    auto update_single_acc = [&](NNUE::Accumulator &new_acc, NNUE::Accumulator &old_acc, std::vector<NNUEIndices> &adds, std::vector<NNUEIndices> &subs) {
        int num_add = adds.size();
        int num_sub = subs.size();

        // Quiets
        if (num_add == 1 && num_sub == 1) {
            addSub(tmp_acc, old_acc, adds[0], subs[0]);
        }
        // Captures
        else if (num_add == 1 && num_sub == 2) {
            addSubSub(tmp_acc, old_acc, adds[0], subs[0], subs[1]);
        }
        // Castling
        else {
            // Note that for second addSub we use tmp_acc because we are updating on top of it rather than resetting our reference
            addSub(tmp_acc, old_acc, adds[0], subs[0]);
            addSub(tmp_acc, tmp_acc, adds[1], subs[1]);
        }
        for (int k = 0; k < L1_SIZE; ++k)
            new_acc.values[0][k] = tmp_acc.values[0][k];

        for (int k = 0; k < L1_SIZE; ++k)
            new_acc.values[1][k] = tmp_acc.values[1][k];

        adds.clear();
        subs.clear();
    };

    // For the bottom-most accumulator we update it outside the loop
    if (i - 1 >= 0) {
        NNUE::Accumulator &new_acc = acc[-(i - 1)];
        NNUE::Accumulator &old_acc = acc[-i];
        std::vector<NNUEIndices> &adds = new_acc.NNUEAdd;
        std::vector<NNUEIndices> &subs = new_acc.NNUESub;

        update_single_acc(new_acc, old_acc, adds, subs);
    }

    // Iterate backwards to the top from the second accumulator we need to update onwards
    for (int j = i - 2; j >= 0; --j) {
        NNUE::Accumulator &new_acc = acc[-j];
        std::vector<NNUEIndices> &adds = new_acc.NNUEAdd;
        std::vector<NNUEIndices> &subs = new_acc.NNUESub;

        update_single_acc(new_acc, tmp_acc, adds, subs);
    }
}

void NNUE::addSub(NNUE::Accumulator &new_acc, NNUE::Accumulator &prev_acc, NNUEIndices add, NNUEIndices sub) {
    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx, blackSubIdx] = sub;
    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub = &net.FTWeights[whiteSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc.values[0][i] = prev_acc.values[0][i] - whiteSub[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub = &net.FTWeights[blackSubIdx * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc.values[1][i] = prev_acc.values[1][i] - blackSub[i] + blackAdd[i];
    }
}

void NNUE::addSubSub(NNUE::Accumulator &new_acc, NNUE::Accumulator &prev_acc, NNUEIndices add, NNUEIndices sub1, NNUEIndices sub2) {

    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx1, blackSubIdx1] = sub1;
    auto [whiteSubIdx2, blackSubIdx2] = sub2;

    auto whiteAdd = &net.FTWeights[whiteAddIdx * L1_SIZE];
    auto whiteSub1 = &net.FTWeights[whiteSubIdx1 * L1_SIZE];
    auto whiteSub2 = &net.FTWeights[whiteSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc.values[0][i] = prev_acc.values[0][i] - whiteSub1[i] - whiteSub2[i] + whiteAdd[i];
    }
    auto blackAdd = &net.FTWeights[blackAddIdx * L1_SIZE];
    auto blackSub1 = &net.FTWeights[blackSubIdx1 * L1_SIZE];
    auto blackSub2 = &net.FTWeights[blackSubIdx2 * L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        new_acc.values[1][i] = prev_acc.values[1][i] - blackSub1[i] - blackSub2[i] + blackAdd[i];
    }
}

int32_t NNUE::ActivateFTAndAffineL1(const int16_t *us, const int16_t *them, const int16_t *weights, const int16_t bias) {
    #if defined(USE_SIMD)
    vepi32 sum  = vec_zero_epi32();
    const vepi16 Zero = vec_zero_epi16();
    const vepi16 One  = vec_set1_epi16(FT_QUANT);
    int weightOffset = 0;
    for (const int16_t *acc : {us, them}) {
        for (int i = 0; i < L1_SIZE; i += CHUNK_SIZE) {
            vepi16 input   = vec_loadu_epi(reinterpret_cast<const vepi16*>(&acc[i]));
            vepi16 weight  = vec_loadu_epi(reinterpret_cast<const vepi16*>(&weights[i + weightOffset]));
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
