#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedSlicedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::SortedRowMajorSlicedEllpack,
                                              TNL::Algorithms::Segments::SortedColumnMajorSlicedEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
