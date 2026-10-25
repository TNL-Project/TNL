#include <TNL/Algorithms/Segments/BiEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedBiEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::SortedRowMajorBiEllpack,
   TNL::Algorithms::Segments::SortedColumnMajorBiEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSortedBiEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
