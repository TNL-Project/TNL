#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Hip.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Hip.h>
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

TYPED_TEST( FillTest, fill_hip )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using HipAllocator = Allocators::Hip< ValueType >;

   HostAllocator hostAllocator;
   HipAllocator hipAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = hipAllocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( hostData, (ValueType) 0, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData, (ValueType) 13, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Hip >( hostData, deviceData, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( hostData[ i ], 13 );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hipAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
}

#include "../main.h"
