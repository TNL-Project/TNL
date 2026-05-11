#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SlicedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorSlicedEllpack,
                                              TNL::Algorithms::Segments::ColumnMajorSlicedEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
