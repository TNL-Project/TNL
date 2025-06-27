#include <iostream>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedEllpack_segments";

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorEllpack = TNL::Algorithms::Segments::SortedRowMajorEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorEllpack = TNL::Algorithms::Segments::SortedColumnMajorEllpack< Device, Index, IndexAllocator >;

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::
      SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedRowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorEllpack >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
