#include <vector>
#include <TNL/Allocators/Host.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/equal.h>
#include <TNL/Algorithms/fill.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class CopyTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( CopyTest, ValueTypes );

TYPED_TEST( CopyTest, copy_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data1 = allocator.allocate( ARRAY_TEST_SIZE );
   ValueType* data2 = allocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, (ValueType) 13, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Host >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   allocator.deallocate( data1, ARRAY_TEST_SIZE );
   allocator.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyWithConversion_host )
{
   using Allocator1 = Allocators::Host< int >;
   using Allocator2 = Allocators::Host< float >;

   Allocator1 allocator1;
   Allocator2 allocator2;
   int* data1 = allocator1.allocate( ARRAY_TEST_SIZE );
   float* data2 = allocator2.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, 13, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Host, float, int, int >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   allocator1.deallocate( data1, ARRAY_TEST_SIZE );
   allocator2.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyArrayToSTLVector_host )
{
   using ValueType = typename TestFixture::ValueType;

   Containers::Array< ValueType > array( ARRAY_TEST_SIZE, 13 );
   std::vector< ValueType > vector1;
   std::vector< ValueType > vector2;
   copy( vector1, array );
   copy( vector2, array.getView() );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector1[ i ], array[ i ] );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector2[ i ], array[ i ] );
}

#include "../main.h"
