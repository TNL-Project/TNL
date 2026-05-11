#include <TNL/Algorithms/Segments/Ellpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_Ellpack_segments";

using MatrixTypes =
   MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorEllpack, TNL::Algorithms::Segments::ColumnMajorEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
