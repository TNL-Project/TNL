#pragma once

#ifdef HAVE_GTEST
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/MemoryOperations.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

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

TYPED_TEST( MemoryOperationsTest, set_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data, (ValueType) 13, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i ++ )
      EXPECT_EQ( data[ i ], 13 );
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

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      ValueType d;
      ASSERT_EQ( cudaMemcpy( &d, &data[ i ], sizeof( ValueType ), cudaMemcpyDeviceToHost ), cudaSuccess );
      EXPECT_EQ( d, i );
      EXPECT_EQ( MemoryOperations< Devices::Cuda >::getElement( &data[ i ] ), i );
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( MemoryOperationsTest, set_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( hostData, (ValueType) 0, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData, (ValueType) 13, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   copy< Devices::Host, Devices::Cuda >( hostData, deviceData, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( hostData[ i ], 13 );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
}

#endif // __CUDACC__
#endif // HAVE_GTEST


#include "../main.h"
