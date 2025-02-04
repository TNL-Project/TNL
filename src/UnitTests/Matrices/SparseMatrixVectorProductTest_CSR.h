#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

template< typename Real, typename Device, typename Index >
struct TestMatrixType
{
   using MatrixType =
      TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types< TestMatrixType< int, TNL::Devices::Host, int >,
                                      TestMatrixType< long, TNL::Devices::Host, int >,
                                      TestMatrixType< float, TNL::Devices::Host, int >,
                                      TestMatrixType< double, TNL::Devices::Host, int >,
                                      TestMatrixType< int, TNL::Devices::Host, long >,
                                      TestMatrixType< long, TNL::Devices::Host, long >,
                                      TestMatrixType< float, TNL::Devices::Host, long >,
                                      TestMatrixType< double, TNL::Devices::Host, long >,
                                      TestMatrixType< std::complex< float >, TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                      ,
                                      TestMatrixType< int, TNL::Devices::Cuda, int >,
                                      TestMatrixType< long, TNL::Devices::Cuda, int >,
                                      TestMatrixType< float, TNL::Devices::Cuda, int >,
                                      TestMatrixType< double, TNL::Devices::Cuda, int >,
                                      TestMatrixType< int, TNL::Devices::Cuda, long >,
                                      TestMatrixType< long, TNL::Devices::Cuda, long >,
                                      TestMatrixType< float, TNL::Devices::Cuda, long >,
                                      TestMatrixType< double, TNL::Devices::Cuda, long >
//,TestMatrixType< TNL::Arithmetics::Complex<float>,   TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                      ,
                                      TestMatrixType< int, TNL::Devices::Hip, int >,
                                      TestMatrixType< long, TNL::Devices::Hip, int >,
                                      TestMatrixType< float, TNL::Devices::Hip, int >,
                                      TestMatrixType< double, TNL::Devices::Hip, int >,
                                      TestMatrixType< int, TNL::Devices::Hip, long >,
                                      TestMatrixType< long, TNL::Devices::Hip, long >,
                                      TestMatrixType< float, TNL::Devices::Hip, long >,
                                      TestMatrixType< double, TNL::Devices::Hip, long >
//,TestMatrixType< TNL::Arithmetics::Complex<float>,   TNL::Devices::Hip, long >
#endif
                                      >;

#include "SparseMatrixVectorProductTest.h"
#include "../main.h"
