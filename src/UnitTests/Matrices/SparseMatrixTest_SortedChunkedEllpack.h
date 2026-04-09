#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedChunkedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::SortedRowMajorChunkedEllpack,
                                              TNL::Algorithms::Segments::SortedColumnMajorChunkedEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
