#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_ChunkedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::RowMajorChunkedEllpack,
   TNL::Algorithms::Segments::ColumnMajorChunkedEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixChunkedEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
