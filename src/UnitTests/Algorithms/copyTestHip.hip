#include <vector>
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Hip.h>
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

TYPED_TEST( CopyTest, copy_hip )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using HipAllocator = Allocators::Hip< ValueType >;

   HostAllocator hostAllocator;
   HipAllocator hipAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* hostData2 = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = hipAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = hipAllocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( hostData, (ValueType) 13, ARRAY_TEST_SIZE );

   copy< Devices::Hip, Devices::Host, ValueType >( deviceData, hostData, ARRAY_TEST_SIZE );
   copy< Devices::Hip, Devices::Hip, ValueType, ValueType >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Hip, ValueType, ValueType >( hostData2, deviceData2, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Host >( hostData, hostData2, ARRAY_TEST_SIZE ) ) );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator.deallocate( hostData2, ARRAY_TEST_SIZE );
   hipAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   hipAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyWithConversions_hip )
{
   using HostAllocator1 = Allocators::Host< int >;
   using HostAllocator2 = Allocators::Host< double >;
   using HipAllocator1 = Allocators::Hip< long >;
   using HipAllocator2 = Allocators::Hip< float >;

   HostAllocator1 hostAllocator1;
   HostAllocator2 hostAllocator2;
   HipAllocator1 hipAllocator1;
   HipAllocator2 hipAllocator2;
   int* hostData = hostAllocator1.allocate( ARRAY_TEST_SIZE );
   double* hostData2 = hostAllocator2.allocate( ARRAY_TEST_SIZE );
   long* deviceData = hipAllocator1.allocate( ARRAY_TEST_SIZE );
   float* deviceData2 = hipAllocator2.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( hostData, 13, ARRAY_TEST_SIZE );

   copy< Devices::Hip, Devices::Host, long, int >( deviceData, hostData, ARRAY_TEST_SIZE );
   copy< Devices::Hip, Devices::Hip, float, long >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Hip, double, float >( hostData2, deviceData2, ARRAY_TEST_SIZE );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( hostData[ i ], hostData2[ i ] );
   hostAllocator1.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator2.deallocate( hostData2, ARRAY_TEST_SIZE );
   hipAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   hipAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyArrayToSTLVector_hip )
{
   using ValueType = typename TestFixture::ValueType;

   Containers::Array< ValueType, Devices::Hip > array( ARRAY_TEST_SIZE, 13 );
   std::vector< ValueType > vector1, vector2;
   copy( vector1, array );
   copy( vector2, array.getView() );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector1[ i ], array.getElement( i ) );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector2[ i ], array.getElement( i ) );
}

#include "../main.h"
