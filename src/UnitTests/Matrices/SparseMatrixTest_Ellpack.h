#include <TNL/Algorithms/Segments/Ellpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_Ellpack_segments";

using MatrixTypes =
   MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorEllpack, TNL::Algorithms::Segments::ColumnMajorEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
