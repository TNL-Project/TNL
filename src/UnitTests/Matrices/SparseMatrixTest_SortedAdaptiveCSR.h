#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedAdaptiveCSR_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::SortedAdaptiveCSR >;

#include "SparseMatrixTest.h"
#include "../main.h"
