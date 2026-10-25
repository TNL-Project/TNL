#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedAdaptiveCSR_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::SortedAdaptiveCSR >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSortedAdaptiveCSR, MatrixTest, MatrixTypes );

#include "../main.h"
