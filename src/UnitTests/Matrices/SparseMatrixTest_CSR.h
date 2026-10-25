#include <TNL/Algorithms/Segments/CSR.h>
#include "SparseMatrixTest_types.h"

constexpr const char* saveAndLoadFileName = "test_SparseMatrixTest_CS_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::CSR >;

#include "SparseMatrixTest.h"
#include "../TypedTestSplit.h"

TNL_INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixCSR, MatrixTest, MatrixTypes );

#include "../main.h"
