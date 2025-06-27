#include <iostream>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedBiEllpack_segments";

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorBiEllpack = TNL::Algorithms::Segments::SortedRowMajorBiEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorBiEllpack = TNL::Algorithms::Segments::SortedColumnMajorBiEllpack< Device, Index, IndexAllocator >;

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::
      SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< TNL::Arithmetics::Complex< float >,
                                TNL::Devices::Cuda,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                ColumnMajorSortedBiEllpack >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedRowMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, SortedColumnMajorBiEllpack >,
   TNL::Matrices::SparseMatrix< TNL::Arithmetics::Complex< float >,
                                TNL::Devices::Hip,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                SortedColumnMajorBiEllpack >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
