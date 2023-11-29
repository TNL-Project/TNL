#include <TNL/Allocators/Host.h>
#include <TNL/Devices/Host.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/fill.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class FillTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( FillTest, ValueTypes );

TYPED_TEST( FillTest, fill_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data, (ValueType) 13, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data[ i ], 13 );
   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

#include "../main.h"
