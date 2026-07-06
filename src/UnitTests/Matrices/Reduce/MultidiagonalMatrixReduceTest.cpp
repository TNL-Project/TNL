// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include "MultidiagonalMatrixReduceTest.h"

TEST( MultidiagonalMatrixReduceTest, reduceRows )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixReduceTest, reduceRows_AutoIdentity )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixReduceTest, reduceAllRows )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixReduceTest, reduceAllRows_AutoIdentity )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceAllRows_AutoIdentity< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixReduceTest, reduceRowsWithArgument )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceRowsWithArgument< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixReduceTest, reduceRowsWithArgumentIf )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_reduceRowsWithArgumentIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}
