#pragma once

#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
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

#ifdef __CUDACC__
TYPED_TEST( MemoryOperationsTest, allocateMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Cuda< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   ASSERT_NE( data, nullptr );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( MemoryOperationsTest, setElement_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Cuda< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      MemoryOperations< Devices::Cuda >::setElement( &data[ i ], (ValueType) i );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ ) {
      ValueType d;
      ASSERT_EQ( cudaMemcpy( &d, &data[ i ], sizeof( ValueType ), cudaMemcpyDeviceToHost ), cudaSuccess );
      EXPECT_EQ( d, i );
      EXPECT_EQ( MemoryOperations< Devices::Cuda >::getElement( &data[ i ] ), i );
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}
#endif  // __CUDACC__

#include "../main.h"
