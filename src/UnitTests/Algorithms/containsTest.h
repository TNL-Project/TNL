#pragma once

#include <TNL/Algorithms/contains.h>
#include <TNL/Containers/Array.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Array >
class ContainsTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ContainsTest is instantiated
using ArrayTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Array< int, Devices::Sequential, short >,
   Array< long, Devices::Sequential, short >,
   Array< double, Devices::Sequential, short >,
   Array< int, Devices::Sequential, int >,
   Array< long, Devices::Sequential, int >,
   Array< double, Devices::Sequential, int >,
   Array< int, Devices::Sequential, long >,
   Array< long, Devices::Sequential, long >,
   Array< double, Devices::Sequential, long >,

   Array< int, Devices::Host, short >,
   Array< long, Devices::Host, short >,
   Array< double, Devices::Host, short >,
   Array< int, Devices::Host, int >,
   Array< long, Devices::Host, int >,
   Array< double, Devices::Host, int >,
   Array< int, Devices::Host, long >,
   Array< long, Devices::Host, long >,
   Array< double, Devices::Host, long >
#elif defined( __CUDACC__ )
   Array< int, Devices::Cuda, short >,
   Array< long, Devices::Cuda, short >,
   Array< double, Devices::Cuda, short >,
   Array< int, Devices::Cuda, int >,
   Array< long, Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   Array< int, Devices::Cuda, long >,
   Array< long, Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >
#elif defined( __HIP__ )
   Array< int, Devices::Hip, short >,
   Array< long, Devices::Hip, short >,
   Array< double, Devices::Hip, short >,
   Array< int, Devices::Hip, int >,
   Array< long, Devices::Hip, int >,
   Array< double, Devices::Hip, int >,
   Array< int, Devices::Hip, long >,
   Array< long, Devices::Hip, long >,
   Array< double, Devices::Hip, long >
#endif
   >;

TYPED_TEST_SUITE( ContainsTest, ArrayTypes );

TYPED_TEST( ContainsTest, contains )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType array;
   array.setSize( 1024 );

   for( int i = 0; i < array.getSize(); i++ )
      array.setElement( i, i % 10 );

   for( int i = 0; i < 10; i++ )
      EXPECT_TRUE( contains( array, i ) );

   for( int i = 10; i < 20; i++ )
      EXPECT_FALSE( contains( array, i ) );
}

TYPED_TEST( ContainsTest, containsOnlyValue )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType array;
   array.setSize( 1024 );

   for( int i = 0; i < array.getSize(); i++ )
      array.setElement( i, i % 10 );

   for( int i = 0; i < 20; i++ )
      EXPECT_FALSE( containsOnlyValue( array, i ) );

   array.setValue( 100 );
   EXPECT_TRUE( containsOnlyValue( array, 100 ) );
}

#include "../main.h"
