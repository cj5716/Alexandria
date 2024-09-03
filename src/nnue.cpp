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

void NNUE::init(const char *file) {

    // open the nn file
    FILE *nn = fopen(file, "rb");

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

void NNUE::update(Accumulator *acc, Position *pos) {

    for (int pov = WHITE; pov <= BLACK; ++pov) {

        // Check if we are able to perform UE
        bool canUE = efficientlyUpdatePov(acc, pos, pov);

        // We could not perform efficient updates, so we refresh the accumulator from scratch
        if (!canUE) {
            // Recalculate the accumulator
            NNUE::Pov_Accumulator &povAccumulator = acc->perspective[pov];
            povAccumulator.accumulate(pos);

            // Reset the add and sub vectors for this accumulator, this will make it "clean" for future updates
            povAccumulator.NNUEAdd.clear();
            povAccumulator.NNUESub.clear();

            // Mark the accumulator as refreshed
            povAccumulator.needsRefresh = false;
        }
    }
}

bool NNUE::efficientlyUpdatePov(NNUE::Accumulator *acc, Position *pos, const int pov) {
    // If we are already clean, no updating required
    if (acc->perspective[pov].isClean()) return true;

    NNUE::Accumulator *cleanAcc = acc;
    int accumulatorsToUpdate = 0;

    // Loop backwards until we find a clean accumulator or need to refresh
    // If we reach the root, we have to refresh (there is nothing below to update against)
    while (cleanAcc != &pos->accumStack[0]) {
        // We need to refresh, so we cannot preform UE
        if (cleanAcc->perspective[pov].needsRefresh) return false;

        // We can update from previous accumulator,
        // just need to check if previous accumulator can be updated
        cleanAcc = cleanAcc - 1;

        // We found a clean accumulator, so we can update from there
        if (cleanAcc->perspective[pov].isClean()) break;

        // We need to update this accumulator as well, so increment the counter
        accumulatorsToUpdate++;
    }

    #if defined(USE_SIMD)
    constexpr int NUM_REGS = sizeof(vepi16) / sizeof(int16_t) * 16; // Equivalent to 16 vepi16s
    #else
    constexpr int NUM_REGS = 128; // Arbitrary number; assume 128-bit vectors
    #endif

    int16_t registers[NUM_REGS];

    for (int offset = 0; offset < L1_SIZE; offset += NUM_REGS) {
        // Load the clean accumulator into the registers
        for (int i = 0; i < NUM_REGS; ++i)
            registers[i] = cleanAcc->perspective[pov].values[offset + i];

        // Loop through the unclean accumulators
        for (int i = accumulatorsToUpdate; i >= 0; --i) {

            NNUE::Pov_Accumulator &currPovAccum = (acc - i)->perspective[pov];
            assert(!currPovAccum.isClean());
            auto adds = currPovAccum.NNUEAdd;
            auto subs = currPovAccum.NNUESub;

            // Assume a quiet move first, with 1 add and 1 sub
            const int16_t *addSlice0 = &net.FTWeights[offset + adds[0] * L1_SIZE];
            const int16_t *subSlice0 = &net.FTWeights[offset + subs[0] * L1_SIZE];
            for (int k = 0; k < NUM_REGS; ++k)
                registers[k] += addSlice0[k] - subSlice0[k];

            // 2 adds mean castling as we cannot move 2 pieces in a move otherwise
            if (adds.size() == 2) {
                const int16_t *addSlice1 = &net.FTWeights[offset + adds[1] * L1_SIZE];
                for (int k = 0; k < NUM_REGS; ++k)
                    registers[k] += addSlice1[k];
            }

            // 2 subs means castling or capture
            if (subs.size() == 2) {
                const int16_t *subSlice1 = &net.FTWeights[offset + subs[1] * L1_SIZE];
                for (int k = 0; k < NUM_REGS; ++k)
                    registers[k] -= subSlice1[k];
            }

            // Store the values from the registers into the accumulators
            int16_t *accumSlice = &currPovAccum.values[offset];
            for (int k = 0; k < NUM_REGS; ++k)
                accumSlice[k] = registers[k];
        }
    }

    for (int i = accumulatorsToUpdate; i >= 0; --i) {
        NNUE::Pov_Accumulator &currPovAccum = (acc - i)->perspective[pov];
        currPovAccum.NNUEAdd.clear();
        currPovAccum.NNUESub.clear();
    }

    return true;

}

void NNUE::Pov_Accumulator::accumulate(Position *pos) {
    for (int i = 0; i < L1_SIZE; i++) {
       values[i] = net.FTBiases[i];
    }

    const bool flip = get_file[KingSQ(pos, pov)] > 3;

    for (int square = 0; square < 64; square++) {
        const bool input = pos->pieces[square] != EMPTY;
        if (!input) continue;
        const auto Idx = GetIndex(pos->pieces[square], square, flip);
        const auto Add = &net.FTWeights[Idx * L1_SIZE];
        for (int j = 0; j < L1_SIZE; j++) {
            values[j] += Add[j];
        }
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

int32_t NNUE::output(const NNUE::Accumulator& board_accumulator, const int stm, const int outputBucket) {
    // this function takes the net output for the current accumulators and returns the eval of the position
    // according to the net
    const int16_t* us;
    const int16_t* them;

    us = board_accumulator.perspective[stm].values.data();
    them = board_accumulator.perspective[stm ^ 1].values.data();

    const int32_t bucketOffset = 2 * L1_SIZE * outputBucket;
    return ActivateFTAndAffineL1(us, them, &net.L1Weights[bucketOffset], net.L1Biases[outputBucket]);
}

void NNUE::accumulate(NNUE::Accumulator& board_accumulator, Position* pos) {
    for(auto& pov_acc : board_accumulator.perspective) {
        pov_acc.accumulate(pos);
    }
}

int NNUE::Pov_Accumulator::GetIndex(const int piece, const int square, bool flip) const {
    constexpr std::size_t COLOR_STRIDE = 64 * 6;
    constexpr std::size_t PIECE_STRIDE = 64;
    const int piecetype = GetPieceType(piece);
    const int pieceColor = Color[piece];
    auto pieceColorPov = pov == WHITE ? pieceColor : (1 ^ pieceColor);
    // Get the final indexes of the updates, accounting for hm
    auto squarePov = pov == WHITE ? (square ^ 0b111'000) : square;
    if(flip) squarePov ^= 0b000'111;
    std::size_t Idx = pieceColorPov * COLOR_STRIDE + piecetype * PIECE_STRIDE + squarePov;
    return Idx;
}
