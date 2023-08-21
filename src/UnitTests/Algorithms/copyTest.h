#pragma once

#include <vector>
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
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

TYPED_TEST( CopyTest, copy_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data1 = allocator.allocate( ARRAY_TEST_SIZE );
   ValueType* data2 = allocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, (ValueType) 13, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Host >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   allocator.deallocate( data1, ARRAY_TEST_SIZE );
   allocator.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyWithConversion_host )
{
   using Allocator1 = Allocators::Host< int >;
   using Allocator2 = Allocators::Host< float >;

   Allocator1 allocator1;
   Allocator2 allocator2;
   int* data1 = allocator1.allocate( ARRAY_TEST_SIZE );
   float* data2 = allocator2.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( data1, 13, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Host, float, int, int >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   allocator1.deallocate( data1, ARRAY_TEST_SIZE );
   allocator2.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyArrayToSTLVector_host )
{
   using ValueType = typename TestFixture::ValueType;

   Containers::Array< ValueType > array( ARRAY_TEST_SIZE, 13 );
   std::vector< ValueType > vector1, vector2;
   copy( vector1, array );
   copy( vector2, array.getView() );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector1[ i ], array[ i ] );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector2[ i ], array[ i ] );
}

#ifdef __CUDACC__
TYPED_TEST( CopyTest, copy_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* hostData2 = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( hostData, (ValueType) 13, ARRAY_TEST_SIZE );

   copy< Devices::Cuda, Devices::Host, ValueType >( deviceData, hostData, ARRAY_TEST_SIZE );
   copy< Devices::Cuda, Devices::Cuda, ValueType, ValueType >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Cuda, ValueType, ValueType >( hostData2, deviceData2, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Host >( hostData, hostData2, ARRAY_TEST_SIZE ) ) );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator.deallocate( hostData2, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyWithConversions_cuda )
{
   using HostAllocator1 = Allocators::Host< int >;
   using HostAllocator2 = Allocators::Host< double >;
   using CudaAllocator1 = Allocators::Cuda< long >;
   using CudaAllocator2 = Allocators::Cuda< float >;

   HostAllocator1 hostAllocator1;
   HostAllocator2 hostAllocator2;
   CudaAllocator1 cudaAllocator1;
   CudaAllocator2 cudaAllocator2;
   int* hostData = hostAllocator1.allocate( ARRAY_TEST_SIZE );
   double* hostData2 = hostAllocator2.allocate( ARRAY_TEST_SIZE );
   long* deviceData = cudaAllocator1.allocate( ARRAY_TEST_SIZE );
   float* deviceData2 = cudaAllocator2.allocate( ARRAY_TEST_SIZE );
   fill< Devices::Host >( hostData, 13, ARRAY_TEST_SIZE );

   copy< Devices::Cuda, Devices::Host, long, int >( deviceData, hostData, ARRAY_TEST_SIZE );
   copy< Devices::Cuda, Devices::Cuda, float, long >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   copy< Devices::Host, Devices::Cuda, double, float >( hostData2, deviceData2, ARRAY_TEST_SIZE );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( hostData[ i ], hostData2[ i ] );
   hostAllocator1.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator2.deallocate( hostData2, ARRAY_TEST_SIZE );
   cudaAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( CopyTest, copyArrayToSTLVector_cuda )
{
   using ValueType = typename TestFixture::ValueType;

   Containers::Array< ValueType, Devices::Cuda > array( ARRAY_TEST_SIZE, 13 );
   std::vector< ValueType > vector1, vector2;
   copy( vector1, array );
   copy( vector2, array.getView() );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector1[ i ], array.getElement( i ) );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( vector2[ i ], array.getElement( i ) );
}
#endif  // __CUDACC__

#include "../main.h"
