#include <iostream>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedSlicedEllpack_segments";

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpack = TNL::Algorithms::Segments::SortedRowMajorSlicedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpack =
   TNL::Algorithms::Segments::SortedColumnMajorSlicedEllpack< Device, Index, IndexAllocator >;

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::
      SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedRowMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorSlicedEllpack >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
