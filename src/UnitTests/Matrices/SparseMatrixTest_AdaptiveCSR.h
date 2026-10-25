#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_AdaptiveCSR_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::AdaptiveCSR >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixAdaptiveCSR, MatrixTest, MatrixTypes );

#include "../main.h"
