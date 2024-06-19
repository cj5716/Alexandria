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