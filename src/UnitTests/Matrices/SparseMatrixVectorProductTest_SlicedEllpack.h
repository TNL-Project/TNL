#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/SlicedEllpackKernel.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

template< typename Real, typename Device, typename Index, TNL::Algorithms::Segments::ElementsOrganization Organization >
struct MatrixAndKernel
{
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using Segments = TNL::Algorithms::Segments::SlicedEllpack< Device_, Index_, IndexAllocator_, Organization, 32 >;

   using MatrixType = TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
   using KernelType = TNL::Algorithms::SegmentsReductionKernels::SlicedEllpackKernel< Index, Device >;
};

// types for which MatrixTest is instantiated
using MatrixAndKernelTypes = ::testing::Types<
   MatrixAndKernel< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< long, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< int, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< long, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< float, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< std::complex< float >, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< long,    TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< float,   TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< double,  TNL::Devices::Host, int,  TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< int,     TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< long,    TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   //MatrixAndKernel< float,   TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Host, long, TNL::Algorithms::Segments::ColumnMajorOrder >
#ifdef __CUDACC__
   ,
   MatrixAndKernel< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< long,    TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< float,   TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< double,  TNL::Devices::Cuda, int,  TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< int,     TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< long,    TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   //MatrixAndKernel< float,   TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::RowMajorOrder >,
   MatrixAndKernel< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< long, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< int, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< long, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< float, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   MatrixAndKernel< TNL::Arithmetics::Complex< float >, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >
#endif
   >;

#include "SparseMatrixVectorProductTest.h"
#include "../main.h"
