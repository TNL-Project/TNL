#include <TNL/Algorithms/Segments/BiEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_BiEllpack_segments";

using MatrixTypes =
   MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorBiEllpack, TNL::Algorithms::Segments::ColumnMajorBiEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixBiEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
