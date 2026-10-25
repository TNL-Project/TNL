#include <TNL/Algorithms/Segments/CSR.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedCSR_segments";

using MatrixTypes = MatrixTypesTemplate< TNL::Algorithms::Segments::SortedCSR >;

#include "SparseMatrixTest.h"
#include "../main.h"
