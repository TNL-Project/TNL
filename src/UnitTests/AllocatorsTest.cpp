#include <TNL/Allocators/Host.h>

#include "gtest/gtest.h"

using namespace TNL;

constexpr int ARRAY_TEST_SIZE = 100;

// test fixture for typed tests
template< typename Value >
class AllocatorsTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( AllocatorsTest, ValueTypes );

TYPED_TEST( AllocatorsTest, Host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // do something useful with the data
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ ) {
      data[ i ] = 0;
      EXPECT_EQ( data[ i ], 0 );
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

#include "main.h"
