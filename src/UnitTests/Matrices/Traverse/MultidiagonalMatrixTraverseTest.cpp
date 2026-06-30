// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include "MultidiagonalMatrixTraverseTest.h"

TEST( MultidiagonalMatrixTraverseTest, forElements_Range )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forElements_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forAllElements )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forAllElements< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forElements_WithIndexArray )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forElements_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forElementsIf )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forElementsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forRows_Range )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forRows_Range< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forAllRows )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forAllRows< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forRows_WithIndexArray )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forRows_WithIndexArray< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}

TEST( MultidiagonalMatrixTraverseTest, forRowsIf )
{
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long > >();
#elif defined( __CUDACC__ )
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long > >();
#elif defined( __HIP__ )
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long > >();
   test_forRowsIf< TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long > >();
#endif
}
