#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_AdaptiveCSR_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::AdaptiveCSR >;

#include "SparseMatrixTest.h"
#include "../main.h"
