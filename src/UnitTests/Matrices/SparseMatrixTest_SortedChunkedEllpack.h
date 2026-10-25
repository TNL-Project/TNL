#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedChunkedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::SortedRowMajorChunkedEllpack,
   TNL::Algorithms::Segments::SortedColumnMajorChunkedEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSortedChunkedEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
