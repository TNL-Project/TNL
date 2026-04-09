#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include "SparseMatrixTest_types.h"

const char* saveAndLoadFileName = "test_SparseMatrixTest_ChunkedEllpack_segments";

using MatrixTypes = MatrixTypesTemplateMixed< TNL::Algorithms::Segments::RowMajorChunkedEllpack,
                                              TNL::Algorithms::Segments::ColumnMajorChunkedEllpack >;

#include "SparseMatrixTest.h"
#include "../main.h"
