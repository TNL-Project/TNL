#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_SlicedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed<
   TNL::Algorithms::Segments::RowMajorSlicedEllpack,
   TNL::Algorithms::Segments::ColumnMajorSlicedEllpack >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixSlicedEllpack, MatrixTest, MatrixTypes );

#include "../main.h"
