#include <TNL/Allocators/Host.h>
#include <TNL/Containers/detail/MemoryOperations.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers::detail;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class MemoryOperationsTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( MemoryOperationsTest, ValueTypes );

TYPED_TEST( MemoryOperationsTest, allocateMemory_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, setElement_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ ) {
      MemoryOperations< Devices::Host >::setElement( data + i, (ValueType) i );
      EXPECT_EQ( data[ i ], i );
      EXPECT_EQ( MemoryOperations< Devices::Host >::getElement( data + i ), i );
   }
   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

#include "../main.h"
