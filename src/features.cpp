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

