#include <TNL/Algorithms/Segments/Ellpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::SortedRowMajorEllpack,
   TNL::Algorithms::Segments::SortedColumnMajorEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSortedEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
