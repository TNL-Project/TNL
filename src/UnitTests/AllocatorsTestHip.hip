#include <TNL/Backend.h>
#include <TNL/Allocators/Hip.h>
#include <TNL/Allocators/HipHost.h>
#include <TNL/Allocators/HipManaged.h>

#include "gtest/gtest.h"

using namespace TNL;

constexpr int ARRAY_TEST_SIZE = 100;

// test fixture for typed tests
template< typename Value >
class AllocatorsTestHip : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( AllocatorsTestHip, ValueTypes );

TYPED_TEST( AllocatorsTestHip, HipHost )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::HipHost< ValueType >;

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

TYPED_TEST( AllocatorsTestHip, HipManaged )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::HipManaged< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // fill data on the device
   ValueType host_data[ ARRAY_TEST_SIZE ];
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      host_data[ i ] = 0;
   Backend::memcpy(
      static_cast< void* >( data ), static_cast< void* >( host_data ), ARRAY_TEST_SIZE, Backend::MemcpyHostToDevice );

   // check values on the host
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data[ i ], 0 );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( AllocatorsTestHip, Hip )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Hip< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // fill data on the device
   ValueType host_data[ ARRAY_TEST_SIZE ];
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      host_data[ i ] = 0;
   Backend::memcpy(
      static_cast< void* >( data ), static_cast< void* >( host_data ), ARRAY_TEST_SIZE, Backend::MemcpyHostToDevice );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

#include "main.h"
