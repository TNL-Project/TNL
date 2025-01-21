#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Containers/Vector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Vector >
class CompressTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
};

// types for which ContainsTest is instantiated
using VectorTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Vector< int, Devices::Sequential, int >,
   Vector< long, Devices::Sequential, int >,
   Vector< double, Devices::Sequential, int >,
   Vector< int, Devices::Sequential, long >,
   Vector< long, Devices::Sequential, long >,
   Vector< double, Devices::Sequential, long >,

   Vector< int, Devices::Host, int >,
   Vector< long, Devices::Host, int >,
   Vector< double, Devices::Host, int >,
   Vector< int, Devices::Host, long >,
   Vector< long, Devices::Host, long >,
   Vector< double, Devices::Host, long >
#elif defined( __CUDACC__ )
   Vector< int, Devices::Cuda, int >,
   Vector< long, Devices::Cuda, int >,
   Vector< double, Devices::Cuda, int >,
   Vector< int, Devices::Cuda, long >,
   Vector< long, Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >
#elif defined( __HIP__ )
   Vector< int, Devices::Hip, int >,
   Vector< long, Devices::Hip, int >,
   Vector< double, Devices::Hip, int >,
   Vector< int, Devices::Hip, long >,
   Vector< long, Devices::Hip, long >,
   Vector< double, Devices::Hip, long >
#endif
   >;

TYPED_TEST_SUITE( CompressTest, VectorTypes );

template< typename VectorType >
void
CompressTest_compressVector()
{
   using IndexType = typename VectorType::IndexType;

   //clang-format off
   //            0  1  2  3  4  5  6  7  8  9 10
   VectorType v{ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
   //clang-format on
   auto compressed = compress< VectorType, VectorType, IndexType >( v );
   EXPECT_EQ( compressed.getSize(), 4 );
   EXPECT_EQ( compressed.getElement( 0 ), 0 );
   EXPECT_EQ( compressed.getElement( 1 ), 5 );
   EXPECT_EQ( compressed.getElement( 2 ), 7 );
   EXPECT_EQ( compressed.getElement( 3 ), 8 );

   auto v_view = v.getView();
   auto compressed2 = compress< VectorType >( 0,
                                              v.getSize(),
                                              [ = ] __cuda_callable__( IndexType idx )
                                              {
                                                 return v_view[ idx ];
                                              } );

   EXPECT_EQ( compressed2.getSize(), 4 );
   EXPECT_EQ( compressed2.getElement( 0 ), 0 );
   EXPECT_EQ( compressed2.getElement( 1 ), 5 );
   EXPECT_EQ( compressed2.getElement( 2 ), 7 );
   EXPECT_EQ( compressed2.getElement( 3 ), 8 );

   auto compressed3 = compress< VectorType >( 4,
                                              v.getSize(),
                                              [ = ] __cuda_callable__( IndexType idx )
                                              {
                                                 return v_view[ idx ];
                                              } );

   EXPECT_EQ( compressed3.getSize(), 3 );
   EXPECT_EQ( compressed3.getElement( 0 ), 5 );
   EXPECT_EQ( compressed3.getElement( 1 ), 7 );
   EXPECT_EQ( compressed3.getElement( 2 ), 8 );
}

TYPED_TEST( CompressTest, compressVector )
{
   CompressTest_compressVector< typename TestFixture::VectorType >();
}

#include "../main.h"
