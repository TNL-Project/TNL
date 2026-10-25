#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedSlicedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::SortedRowMajorSlicedEllpack,
   TNL::Algorithms::Segments::SortedColumnMajorSlicedEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSortedSlicedEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
