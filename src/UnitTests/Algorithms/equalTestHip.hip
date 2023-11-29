#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Hip.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Hip.h>
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

TYPED_TEST( EqualTest, equal_hip )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using HipAllocator = Allocators::Hip< ValueType >;

   HostAllocator hostAllocator;
   HipAllocator hipAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = hipAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = hipAllocator.allocate( ARRAY_TEST_SIZE );

   fill< Devices::Host >( hostData, (ValueType) 7, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData, (ValueType) 8, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData2, (ValueType) 9, ARRAY_TEST_SIZE );

   EXPECT_FALSE( ( equal< Devices::Host, Devices::Hip >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Hip, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Hip, Devices::Hip >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   fill< Devices::Hip >( deviceData, (ValueType) 7, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData2, (ValueType) 7, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Hip >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Hip, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Hip, Devices::Hip >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hipAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   hipAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( EqualTest, equalWithConversions_hip )
{
   using HostAllocator = Allocators::Host< int >;
   using HipAllocator1 = Allocators::Hip< float >;
   using HipAllocator2 = Allocators::Hip< double >;

   HostAllocator hostAllocator;
   HipAllocator1 hipAllocator1;
   HipAllocator2 hipAllocator2;
   int* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   float* deviceData = hipAllocator1.allocate( ARRAY_TEST_SIZE );
   double* deviceData2 = hipAllocator2.allocate( ARRAY_TEST_SIZE );

   fill< Devices::Host >( hostData, 7, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData, (float) 8, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData2, (double) 9, ARRAY_TEST_SIZE );

   EXPECT_FALSE( ( equal< Devices::Host, Devices::Hip >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Hip, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Hip, Devices::Hip >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   fill< Devices::Hip >( deviceData, (float) 7, ARRAY_TEST_SIZE );
   fill< Devices::Hip >( deviceData2, (double) 7, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Hip >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Hip, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Hip, Devices::Hip >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hipAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   hipAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

#include "../main.h"
