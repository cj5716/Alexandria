#include <cstdint>
#include "incbin/incbin.h"

// Macro to embed the position feature file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gFEATUREData[];  // a pointer to the embedded data
//     const unsigned char *const gFEATUREEnd;     // a marker to the end
//     const unsigned int         gFEATURESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER)
INCBIN(FEATURE, FEATUREFILE);
#else
const unsigned char gFEATUREData[1] = {};
const unsigned char* const gFEATUREEnd = &gFEATUREData[1];
const unsigned int gFEATURESize = 1;
#endif

FeatureNet featureNet;

void FeatureNet::init(const char* file) {

    // open the nn file
    FILE* fnn = fopen(file, "rb");

    // if it's not invalid read the config values from it
    if (fnn) {
        // initialize an accumulator for every input of the second layer
        size_t read = 0;
        const size_t fileSize = sizeof(FeatureNet);
        const size_t objectsExpected = fileSize / sizeof(int16_t);

        read += fread(FeatureWeights, sizeof(int16_t), NUM_INPUTS * NUM_FEATURES, fnn);
        read += fread(FeatureBiases , sizeof(int16_t), NUM_FEATURES, fnn);

        if (read != objectsExpected) {
            std::cout << "Error loading the net, aborting ";
            std::cout << "Expected " << objectsExpected << " shorts, got " << read << "\n";
            exit(1);
        }

        // after reading the config we can close the file
        fclose(fnn);

    } else {
        // if we don't find the nnue file we use the net embedded in the exe
        uint64_t memoryIndex = 0;
        std::memcpy(FeatureWeights, &gFEATUREData[memoryIndex], NUM_INPUTS * NUM_FEATURES * sizeof(int16_t));
        memoryIndex += NUM_INPUTS * NUM_FEATURES * sizeof(int16_t);
        std::memcpy(FeatureBiases, &gFEATUREData[memoryIndex], NUM_FEATURES * sizeof(int16_t));
    }
}

void FeatureAccumulator::Accumulate(Position *pos) {
    for (int i = 0; i < NUM_FEATURES; ++i) {
        values[0][i] = featureNet.FeatureWeights[i];
        values[1][i] = featureNet.FeatureWeights[i];
    }

    for (int i = 0; i < 64; i++) {
        bool input = pos->pieces[i] != EMPTY;
        if (!input) continue;

        auto [whiteIdx, blackIdx] = GetIndex(pos->pieces[i], i);
        auto whiteAdd = &featureNet.FeatureWeights[whiteIdx * NUM_FEATURES];
        auto blackAdd = &featureNet.FeatureWeights[blackIdx * NUM_FEATURES];

        for (int j = 0; j < NUM_FEATURES; ++j) {
            values[0][j] += whiteAdd[j];
        }
        for (int j = 0; j < NUM_FEATURES; ++j) {
            values[1][j] += blackAdd[j];
        }
    }
}

uint64_t FeatureAccumulator::GetFeatureHash(const int side) {
    uint64_t hash = 0;
    for (int i = 0; i < NUM_FEATURES; ++i) {
        hash |= uint64_t(values[side][i]) << i;
    }

    for (int i = 0; i < NUM_FEATURES; ++i) {
        hash |= uint64_t(values[side ^ 1][i]) << (i + NUM_FEATURES);
    }

    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33;

    return hash;
}

void UpdateFeatureAccumulator(FeatureAccumulator *acc) {
    int adds = acc->FeatureAdd.size();
    int subs = acc->FeatureSub.size();

    if (adds == 0 && subs == 0)
        return;

    if (!(acc - 1)->FeatureAdd.empty() && !(acc - 1)->FeatureSub.empty())
        update(acc - 1);

    // Quiets
    if (adds == 1 && subs == 1) {
        FeatureAddSub(acc, acc - 1, acc->FeatureAdd[0], acc->FeatureSub[0]);
    }
    // Captures
    else if (adds == 1 && subs == 2) {
        FeatureAddSubSub(acc, acc - 1, acc->FeatureAdd[0], acc->FeatureSub[0], acc->FeatureSub[1]);
    }
    // Castling
    else {
        FeatureAddSub(acc, acc - 1, acc->FeatureAdd[0], acc->FeatureSub[0]);
        FeatureAddSub(acc, acc, acc->FeatureAdd[1], acc->FeatureSub[1]);
        // Note that for second addSub, we put acc instead of acc - 1 because we are updating on top of
        // the half-updated accumulator
    }
    // Reset the add and sub vectors
    acc->FeatureAdd.clear();
    acc->FeatureSub.clear();
}

void FeatureAddSub(FeatureAccumulator *new_acc, FeatureAccumulator *prev_acc, FeatureIndices add, FeatureIndices sub) {
    auto [whiteAddIdx, blackAddIdx] = add;
    auto [whiteSubIdx, blackSubIdx] = sub;

    auto whiteAdd = &featureNet.FeatureWeights[whiteAddIdx * NUM_FEATURES];
    auto whiteSub = &featureNet.FeatureWeights[whiteSubIdx * NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) {
        new_acc->values[0][i] = prev_acc->values[0][i] - whiteSub[i] + whiteAdd[i];
    }

    auto blackAdd = &featureNet.FeatureWeights[blackAddIdx * NUM_FEATURES];
    auto blackSub = &featureNet.FeatureWeights[blackSubIdx * NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) {
        new_acc->values[1][i] = prev_acc->values[1][i] - blackSub[i] + blackAdd[i];
    }
}

void FeatureAddSubSub(FeatureAccumulator *new_acc, FeatureAccumulator *prev_acc, FeatureIndices add, FeatureIndices sub1, FeatureIndices sub2) {
    auto [whiteAddIdx , blackAddIdx ] = add;
    auto [whiteSubIdx1, blackSubIdx1] = sub1;
    auto [whiteSubIdx2, blackSubIdx2] = sub2;

    auto whiteAdd  = &featureNet.FeatureWeights[whiteAddIdx  * NUM_FEATURES];
    auto whiteSub1 = &featureNet.FeatureWeights[whiteSubIdx1 * NUM_FEATURES];
    auto whiteSub2 = &featureNet.FeatureWeights[whiteSubIdx2 * NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) {
        new_acc->values[0][i] = prev_acc->values[0][i] - whiteSub1[i] - whiteSub2[i] + whiteAdd[i];
    }

    auto blackAdd  = &featureNet.FeatureWeights[blackAddIdx  * NUM_FEATURES];
    auto blackSub1 = &featureNet.FeatureWeights[blackSubIdx1 * NUM_FEATURES];
    auto blackSub2 = &featureNet.FeatureWeights[blackSubIdx2 * NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) {
        new_acc->values[1][i] = prev_acc->values[1][i] - blackSub1[i] - blackSub2[i] + blackAdd[i];
    }
}

uint64_t GetFeatureHash(Position *pos) {
    UpdateFeatureAccumulator(&pos->featureTop());
    return pos->featureTop().GetFeatureHash(pos->side);
}