#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

template< typename Real, typename Device, typename Index >
struct MatrixAndKernel
{
   using MatrixType =
      TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >;
   using KernelType = TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >;
};

// types for which MatrixTest is instantiated
using MatrixAndKernelTypes = ::testing::Types< MatrixAndKernel< int, TNL::Devices::Host, int >,
                                               MatrixAndKernel< long, TNL::Devices::Host, int >,
                                               MatrixAndKernel< float, TNL::Devices::Host, int >,
                                               MatrixAndKernel< double, TNL::Devices::Host, int >,
                                               MatrixAndKernel< int, TNL::Devices::Host, long >,
                                               MatrixAndKernel< long, TNL::Devices::Host, long >,
                                               MatrixAndKernel< float, TNL::Devices::Host, long >,
                                               MatrixAndKernel< double, TNL::Devices::Host, long >,
                                               MatrixAndKernel< std::complex< float >, TNL::Devices::Host, long >
#ifdef __CUDACC__
                                               ,
                                               MatrixAndKernel< int, TNL::Devices::Cuda, int >,
                                               MatrixAndKernel< long, TNL::Devices::Cuda, int >,
                                               MatrixAndKernel< float, TNL::Devices::Cuda, int >,
                                               MatrixAndKernel< double, TNL::Devices::Cuda, int >,
                                               MatrixAndKernel< int, TNL::Devices::Cuda, long >,
                                               MatrixAndKernel< long, TNL::Devices::Cuda, long >,
                                               MatrixAndKernel< float, TNL::Devices::Cuda, long >,
                                               MatrixAndKernel< double, TNL::Devices::Cuda, long >,
                                               MatrixAndKernel< TNL::Arithmetics::Complex< float >, TNL::Devices::Cuda, long >
#endif
                                               >;

#include "SparseMatrixVectorProductTest.h"
#include "../main.h"
