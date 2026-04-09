#include <TNL/Algorithms/Segments/Ellpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::SortedRowMajorEllpack,
                                              TNL::Algorithms::Segments::SortedColumnMajorEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
