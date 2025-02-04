#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/BiEllpackKernel.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

template< typename Real, typename Device, typename Index, TNL::Algorithms::Segments::ElementsOrganization Organization >
struct TestMatrixType
{
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using Segments = TNL::Algorithms::Segments::BiEllpack< Device_, Index_, IndexAllocator_, Organization >;

   using MatrixType = TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TestMatrixType< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< long, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< int, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< long, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< float, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< double, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< std::complex< float >, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< double,  TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< int,     TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< double, TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >
#if defined( __CUDACC__ )
   ,
   TestMatrixType< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< double,  TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< int,     TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< long, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< int, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< long, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< float, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >
//,TestMatrixType< TNL::Arithmetics::Complex<float>,   TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >
#elif defined( __HIP__ )
   ,
   TestMatrixType< int, TNL::Devices::Hip, int, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Hip, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Hip, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< double,  TNL::Devices::Hip, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< int,     TNL::Devices::Hip, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< long,    TNL::Devices::Hip, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //TestMatrixType< float,   TNL::Devices::Hip, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< double, TNL::Devices::Hip, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TestMatrixType< int, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< long, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< float, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< int, TNL::Devices::Hip, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< long, TNL::Devices::Hip, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< float, TNL::Devices::Hip, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TestMatrixType< double, TNL::Devices::Hip, long, TNL::Algorithms::Segments::ColumnMajorOrder >
//,TestMatrixType< TNL::Arithmetics::Complex<float>,   TNL::Devices::Hip, long, TNL::Algorithms::Segments::ColumnMajorOrder >
#endif
   >;

#include "SparseMatrixVectorProductTest.h"
#include "../main.h"
