#include <TNL/Algorithms/Segments/BiEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_BiEllpack_segments";

using MatrixTypes =
   MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorBiEllpack, TNL::Algorithms::Segments::ColumnMajorBiEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
