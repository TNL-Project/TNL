#include <iostream>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedChunkedEllpack_segments";

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorChunkedEllpack = TNL::Algorithms::Segments::SortedRowMajorChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorChunkedEllpack =
   TNL::Algorithms::Segments::SortedColumnMajorChunkedEllpack< Device, Index, IndexAllocator >;

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::
      SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedRowMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorChunkedEllpack >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
