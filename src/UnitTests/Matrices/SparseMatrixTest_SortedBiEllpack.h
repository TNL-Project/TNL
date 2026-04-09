#include <TNL/Algorithms/Segments/BiEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedBiEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::SortedRowMajorBiEllpack,
                                              TNL::Algorithms::Segments::SortedColumnMajorBiEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
