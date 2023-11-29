#include <gtest/gtest.h>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Devices/Host.h>
#include "LambdaMatrixTest.hpp"
#include <iostream>

template< typename Real, typename Device, typename Index >
struct LambdaMatrixParameters
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
};

// test fixture for typed tests
template< typename Matrix >
class LambdaMatrixTest : public ::testing::Test
{
protected:
   using LambdaMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using LambdaMatrixTypes = ::testing::Types< LambdaMatrixParameters< int, TNL::Devices::Host, int >,
                                            LambdaMatrixParameters< long, TNL::Devices::Host, int >,
                                            LambdaMatrixParameters< float, TNL::Devices::Host, int >,
                                            LambdaMatrixParameters< double, TNL::Devices::Host, int >,
                                            LambdaMatrixParameters< int, TNL::Devices::Host, long >,
                                            LambdaMatrixParameters< long, TNL::Devices::Host, long >,
                                            LambdaMatrixParameters< float, TNL::Devices::Host, long >,
                                            LambdaMatrixParameters< double, TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                            ,
                                            LambdaMatrixParameters< int, TNL::Devices::Cuda, int >,
                                            LambdaMatrixParameters< long, TNL::Devices::Cuda, int >,
                                            LambdaMatrixParameters< float, TNL::Devices::Cuda, int >,
                                            LambdaMatrixParameters< double, TNL::Devices::Cuda, int >,
                                            LambdaMatrixParameters< int, TNL::Devices::Cuda, long >,
                                            LambdaMatrixParameters< long, TNL::Devices::Cuda, long >,
                                            LambdaMatrixParameters< float, TNL::Devices::Cuda, long >,
                                            LambdaMatrixParameters< double, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                            ,
                                            LambdaMatrixParameters< int, TNL::Devices::Hip, int >,
                                            LambdaMatrixParameters< long, TNL::Devices::Hip, int >,
                                            LambdaMatrixParameters< float, TNL::Devices::Hip, int >,
                                            LambdaMatrixParameters< double, TNL::Devices::Hip, int >,
                                            LambdaMatrixParameters< int, TNL::Devices::Hip, long >,
                                            LambdaMatrixParameters< long, TNL::Devices::Hip, long >,
                                            LambdaMatrixParameters< float, TNL::Devices::Hip, long >,
                                            LambdaMatrixParameters< double, TNL::Devices::Hip, long >
#endif
                                            >;

TYPED_TEST_SUITE( LambdaMatrixTest, LambdaMatrixTypes );

TYPED_TEST( LambdaMatrixTest, Constructors )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_Constructors< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, setDimensionsTest )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_SetDimensions< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, getCompressedRowLengthsTest )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_GetCompressedRowLengths< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, getElementTest )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_GetElement< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, forRowsTest )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_ForRows< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, vectorProductTest )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_VectorProduct< LambdaMatrixParametersType >();
}

TYPED_TEST( LambdaMatrixTest, reduceRows )
{
   using LambdaMatrixParametersType = typename TestFixture::LambdaMatrixType;

   test_reduceRows< LambdaMatrixParametersType >();
}

#include "../main.h"
