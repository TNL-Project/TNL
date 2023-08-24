#include <TNL/Allocators/Host.h>
#include <TNL/Devices/Host.h>
#include <TNL/Algorithms/equal.h>
#include <TNL/Algorithms/fill.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class EqualTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( EqualTest, ValueTypes );

TYPED_TEST( EqualTest, equal_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data1 = allocator.allocate( ARRAY_TEST_SIZE );
   ValueType* data2 = allocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, (ValueType) 7, ARRAY_TEST_SIZE );
   fill< Devices::Host >( data2, (ValueType) 0, ARRAY_TEST_SIZE );
   EXPECT_FALSE( ( equal< Devices::Host, Devices::Host >( data1, data2, ARRAY_TEST_SIZE ) ) );
   fill< Devices::Host >( data2, (ValueType) 7, ARRAY_TEST_SIZE );
   EXPECT_TRUE( ( equal< Devices::Host, Devices::Host >( data1, data2, ARRAY_TEST_SIZE ) ) );
   allocator.deallocate( data1, ARRAY_TEST_SIZE );
   allocator.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( EqualTest, equalWithConversion_host )
{
   using Allocator1 = Allocators::Host< int >;
   using Allocator2 = Allocators::Host< float >;

   Allocator1 allocator1;
   Allocator2 allocator2;
   int* data1 = allocator1.allocate( ARRAY_TEST_SIZE );
   float* data2 = allocator2.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, 7, ARRAY_TEST_SIZE );
   fill< Devices::Host >( data2, (float) 0.0, ARRAY_TEST_SIZE );
   EXPECT_FALSE( ( equal< Devices::Host, Devices::Host >( data1, data2, ARRAY_TEST_SIZE ) ) );
   fill< Devices::Host >( data2, (float) 7.0, ARRAY_TEST_SIZE );
   EXPECT_TRUE( ( equal< Devices::Host, Devices::Host >( data1, data2, ARRAY_TEST_SIZE ) ) );
   allocator1.deallocate( data1, ARRAY_TEST_SIZE );
   allocator2.deallocate( data2, ARRAY_TEST_SIZE );
}

#include "../main.h"
