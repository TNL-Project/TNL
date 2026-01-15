#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_Ellpack_segments";

template< typename Device, typename Index, typename IndexAllocator >
using RowMajorEllpack = TNL::Algorithms::Segments::RowMajorEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorEllpack = TNL::Algorithms::Segments::ColumnMajorEllpack< Device, Index, IndexAllocator >;

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorEllpack >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
