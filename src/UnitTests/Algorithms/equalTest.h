#pragma once

#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
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

#ifdef __CUDACC__
TYPED_TEST( EqualTest, equal_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = cudaAllocator.allocate( ARRAY_TEST_SIZE );

   fill< Devices::Host >( hostData, (ValueType) 7, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData, (ValueType) 8, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData2, (ValueType) 9, ARRAY_TEST_SIZE );

   EXPECT_FALSE( ( equal< Devices::Host, Devices::Cuda >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Cuda, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Cuda, Devices::Cuda >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   fill< Devices::Cuda >( deviceData, (ValueType) 7, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData2, (ValueType) 7, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Cuda >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Cuda, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Cuda, Devices::Cuda >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( EqualTest, equalWithConversions_cuda )
{
   using HostAllocator = Allocators::Host< int >;
   using CudaAllocator1 = Allocators::Cuda< float >;
   using CudaAllocator2 = Allocators::Cuda< double >;

   HostAllocator hostAllocator;
   CudaAllocator1 cudaAllocator1;
   CudaAllocator2 cudaAllocator2;
   int* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   float* deviceData = cudaAllocator1.allocate( ARRAY_TEST_SIZE );
   double* deviceData2 = cudaAllocator2.allocate( ARRAY_TEST_SIZE );

   fill< Devices::Host >( hostData, 7, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData, (float) 8, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData2, (double) 9, ARRAY_TEST_SIZE );

   EXPECT_FALSE( ( equal< Devices::Host, Devices::Cuda >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Cuda, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_FALSE( ( equal< Devices::Cuda, Devices::Cuda >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   fill< Devices::Cuda >( deviceData, (float) 7, ARRAY_TEST_SIZE );
   fill< Devices::Cuda >( deviceData2, (double) 7, ARRAY_TEST_SIZE );

   EXPECT_TRUE( ( equal< Devices::Host, Devices::Cuda >( hostData, deviceData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Cuda, Devices::Host >( deviceData, hostData, ARRAY_TEST_SIZE ) ) );
   EXPECT_TRUE( ( equal< Devices::Cuda, Devices::Cuda >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ) );

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}
#endif  // __CUDACC__

#include "../main.h"
