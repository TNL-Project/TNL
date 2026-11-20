// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>    // std::sort
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint32_t
#include <functional>   // std::greater, std::less
#include <random>       // std::mt19937
#include <stdexcept>    // std::invalid_argument
#include <type_traits>  // std::is_signed_v, std::is_same_v
#include <vector>

#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/pivotPartition.h>
#include <TNL/Containers/Array.h>

#include "gtest/gtest.h"

#include "../DeviceMemory.h"

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Containers;

// payload for the regression test of the equal-bucket contract:
// elements with the same key are comparison-equivalent, but they carry
// distinct ids which must survive the partitioning
struct KeyPayload
{
   int key;
   int id;

   constexpr bool
   operator==( const KeyPayload& other ) const
   {
      return key == other.key && id == other.id;
   }
};

// compares only the keys, so many distinct payloads are comparison-equivalent
struct KeyPayloadLess
{
   [[nodiscard]] __cuda_callable__
   constexpr bool
   operator()( const KeyPayload& lhs, const KeyPayload& rhs ) const
   {
      return lhs.key < rhs.key;
   }
};

// value type larger than the shared-memory staging budget of a whole tile,
// which forces the GPU implementation to take the direct-global scatter path
struct LargeValue
{
   int key;
   int id;
   // padding to push the size over the per-block shared-memory budget
   unsigned char padding[ 248 ];

   constexpr bool
   operator==( const LargeValue& other ) const
   {
      return key == other.key && id == other.id;
   }
};

// a tile of 1024 such elements takes 256 KiB, which exceeds the per-block
// shared-memory budget of any GPU
static_assert( sizeof( LargeValue ) == 256, "unexpected size of LargeValue" );

// creates a large value with deterministic padding derived from (key, id)
inline LargeValue
makeLargeValue( int key, int id )
{
   LargeValue value{};
   value.key = key;
   value.id = id;
   for( int i = 0; i < 248; i++ )
      value.padding[ i ] = static_cast< unsigned char >( ( key + id + i ) % 251 );
   return value;
}

// compares only the keys, like KeyPayloadLess
struct LargeValueLess
{
   [[nodiscard]] __cuda_callable__
   constexpr bool
   operator()( const LargeValue& lhs, const LargeValue& rhs ) const
   {
      return lhs.key < rhs.key;
   }
};

// value type small enough for the shared-memory staging of a whole tile,
// with a payload that must survive the byte-wise copies through the shared
// buffer
struct MediumValue
{
   int key;
   int id;
   unsigned char padding[ 24 ];

   constexpr bool
   operator==( const MediumValue& other ) const
   {
      return key == other.key && id == other.id;
   }
};

// a tile of 1024 such elements takes 32 KiB, which fits the default
// per-block shared-memory budget
static_assert( sizeof( MediumValue ) == 32, "unexpected size of MediumValue" );

// creates a medium value with deterministic padding derived from (key, id)
inline MediumValue
makeMediumValue( int key, int id )
{
   MediumValue value{};
   value.key = key;
   value.id = id;
   for( int i = 0; i < 24; i++ )
      value.padding[ i ] = static_cast< unsigned char >( ( key + id + i ) % 251 );
   return value;
}

struct MediumValueLess
{
   [[nodiscard]] __cuda_callable__
   constexpr bool
   operator()( const MediumValue& lhs, const MediumValue& rhs ) const
   {
      return lhs.key < rhs.key;
   }
};

// copies an array-like object into a std::vector on the host
template< typename Array >
std::vector< std::remove_const_t< typename Array::ValueType > >
copyToHost( const Array& array )
{
   std::vector< std::remove_const_t< typename Array::ValueType > > vec;
   TNL::Algorithms::copy( vec, array );
   return vec;
}

// verifies the common partitioning oracle on a host copy of the output:
// the boundaries are ordered and every bucket satisfies its predicate
template< typename Index, typename Value, typename Compare >
void
verifyBuckets(
   const std::vector< Value >& output,
   const PivotPartitionResult< Index >& result,
   const Value& pivot,
   const Compare& compare )
{
   const Index size = static_cast< Index >( output.size() );

   // the boundaries must satisfy 0 <= lessEnd <= greaterBegin <= size
   if constexpr( std::is_signed_v< Index > ) {
      EXPECT_GE( result.lessEnd, 0 );
   }
   ASSERT_LE( result.lessEnd, result.greaterBegin ) << "lessEnd must not exceed greaterBegin";
   ASSERT_LE( result.greaterBegin, size ) << "greaterBegin must not exceed the output size";

   for( Index i = 0; i < result.lessEnd; i++ ) {
      EXPECT_TRUE( compare( output[ i ], pivot ) ) << "unexpected element not less than the pivot at index " << i;
   }
   for( Index i = result.lessEnd; i < result.greaterBegin; i++ ) {
      EXPECT_TRUE( ! compare( output[ i ], pivot ) && ! compare( pivot, output[ i ] ) ) << "unexpected element not equivalent to the pivot at index " << i;
   }
   for( Index i = result.greaterBegin; i < size; i++ ) {
      EXPECT_TRUE( compare( pivot, output[ i ] ) ) << "unexpected element not greater than the pivot at index " << i;
   }
}

// verifies that the output is a permutation of the input;
// the order within the buckets is not checked - the operation is not stable
template< typename Value >
void
verifyPermutation( const std::vector< Value >& output, const std::vector< Value >& input )
{
   std::vector< Value > sortedOutput = output;
   std::vector< Value > sortedInput = input;
   std::sort( sortedOutput.begin(), sortedOutput.end() );
   std::sort( sortedInput.begin(), sortedInput.end() );
   EXPECT_EQ( sortedOutput, sortedInput );
}

// verifies the payload-preservation oracle of the equal bucket:
// the buckets satisfy their predicates, every input payload occurs exactly
// once in the output, and no payload is reconstructed from the pivot;
// the value type must provide comparable `key` and `id` members
template< typename Index, typename Value, typename Compare >
void
verifyPayloadPartition(
   const std::vector< Value >& output,
   const std::vector< Value >& input,
   const PivotPartitionResult< Index >& result,
   const Value& pivot,
   const Compare& compare )
{
   verifyBuckets( output, result, pivot, compare );

   // every input id must occur exactly once in the output and no element
   // equivalent to the pivot may be reconstructed from the pivot
   std::vector< int > idCounts( input.size(), 0 );
   for( const Value& element : output ) {
      EXPECT_NE( element.id, pivot.id ) << "an element equivalent to the pivot was reconstructed from the pivot";
      ASSERT_GE( element.id, 0 );
      ASSERT_LT( element.id, static_cast< int >( input.size() ) );
      ++idCounts[ element.id ];
   }
   for( const int count : idCounts ) {
      EXPECT_EQ( count, 1 ) << "each input payload must occur exactly once in the output";
   }

   // the multiset of (key, id) pairs must be unchanged
   const auto byKeyAndId = []( const Value& lhs, const Value& rhs )
   {
      return lhs.key != rhs.key ? lhs.key < rhs.key : lhs.id < rhs.id;
   };
   std::vector< Value > sortedOutput = output;
   std::vector< Value > sortedInput = input;
   std::sort( sortedOutput.begin(), sortedOutput.end(), byKeyAndId );
   std::sort( sortedInput.begin(), sortedInput.end(), byKeyAndId );
   EXPECT_EQ( sortedOutput, sortedInput );
}

// runs one directed test case through both public overloads and verifies
// the common oracle, the expected boundaries and that the input stays
// unchanged
template< typename ArrayType, typename Compare = std::less<> >
void
verifyDirectedCase(
   const std::vector< typename ArrayType::ValueType >& input,
   const typename ArrayType::ValueType& pivot,
   const Compare& compare = Compare{} )
{
   using Index = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   // expected boundaries computed independently of the implementation
   Index expectedLessEnd = 0;
   Index expectedEqual = 0;
   for( const ValueType& value : input ) {
      if( compare( value, pivot ) )
         ++expectedLessEnd;
      else if( ! compare( pivot, value ) )
         ++expectedEqual;
   }
   const Index expectedGreaterBegin = expectedLessEnd + expectedEqual;

   // out-of-place overload
   ArrayType inputArray( input );
   ArrayType outputArray( static_cast< Index >( input.size() ) );
   const auto result = pivotPartition( inputArray, outputArray, pivot, compare );
   verifyBuckets( copyToHost( outputArray ), result, pivot, compare );
   verifyPermutation( copyToHost( outputArray ), input );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";
   EXPECT_EQ( result.lessEnd, expectedLessEnd );
   EXPECT_EQ( result.greaterBegin, expectedGreaterBegin );

   // in-place overload
   ArrayType dataArray( input );
   const auto inPlaceResult = pivotPartition( dataArray, pivot, compare );
   verifyBuckets( copyToHost( dataArray ), inPlaceResult, pivot, compare );
   verifyPermutation( copyToHost( dataArray ), input );
   EXPECT_EQ( inPlaceResult.lessEnd, expectedLessEnd );
   EXPECT_EQ( inPlaceResult.greaterBegin, expectedGreaterBegin );
}

// test fixture for typed tests
template< typename Array >
class PivotPartitionTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which PivotPartitionTest is instantiated
using ArrayTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Array< int, Devices::Sequential, int >,
   Array< double, Devices::Sequential, long >,
   Array< int, Devices::Host, int >,
   Array< double, Devices::Host, long >,
   Array< std::size_t, Devices::Host, std::size_t >
#elif defined( __CUDACC__ )
   // reduced covering set: one integer, one floating-point and one 64-bit
   // index type
   Array< int, Devices::Cuda, int >,
   Array< double, Devices::Cuda, long >,
   Array< std::size_t, Devices::Cuda, std::size_t >,
   // a narrow index type, which exercises the widening of the block counts
   Array< int, Devices::Cuda, short >
#elif defined( __HIP__ )
   Array< int, Devices::Hip, int >,
   Array< double, Devices::Hip, long >,
   Array< std::size_t, Devices::Hip, std::size_t >,
   // a narrow index type, which exercises the widening of the block counts
   Array< int, Devices::Hip, short >
#endif
   >;

TYPED_TEST_SUITE( PivotPartitionTest, ArrayTypes );

TYPED_TEST( PivotPartitionTest, empty )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ValueType = typename ArrayType::ValueType;
   verifyDirectedCase< ArrayType >( std::vector< ValueType >{}, ValueType{} );
}

TYPED_TEST( PivotPartitionTest, singleElementLess )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1 }, 5 );
}

TYPED_TEST( PivotPartitionTest, singleElementEqual )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 5 }, 5 );
}

TYPED_TEST( PivotPartitionTest, singleElementGreater )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 10 }, 5 );
}

TYPED_TEST( PivotPartitionTest, allEqual )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 7, 7, 7, 7, 7 }, 7 );
}

TYPED_TEST( PivotPartitionTest, allLess )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 2, 3, 4 }, 10 );
}

TYPED_TEST( PivotPartitionTest, allGreater )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 2, 3, 4 }, 0 );
}

TYPED_TEST( PivotPartitionTest, allDistinct )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 5 );
}

TYPED_TEST( PivotPartitionTest, alreadyPartitioned )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 5 );
}

TYPED_TEST( PivotPartitionTest, reverseSorted )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }, 5 );
}

TYPED_TEST( PivotPartitionTest, alternatingBuckets )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 5, 9, 2, 5, 8, 0, 5, 7, 3 }, 5 );
}

TYPED_TEST( PivotPartitionTest, customComparator )
{
   using ArrayType = typename TestFixture::ArrayType;
   // descending order: the first bucket holds the elements greater than the pivot
   verifyDirectedCase< ArrayType >( { 1, 9, 5, 2, 8, 5, 0, 7, 5, 3 }, 5, std::greater<>{} );
}

TYPED_TEST( PivotPartitionTest, pivotAbsent )
{
   using ArrayType = typename TestFixture::ArrayType;
   verifyDirectedCase< ArrayType >( { 1, 3, 7, 9 }, 5 );
}

TYPED_TEST( PivotPartitionTest, arrayViews )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   const std::vector< ValueType > input{ 1, 9, 5, 2, 8, 5, 0, 7, 5, 3 };
   const ValueType pivot = 5;

   // out-of-place overload with a const input view and an output view
   ArrayType inputArray( input );
   ArrayType outputArray( static_cast< IndexType >( input.size() ) );
   auto inputView = inputArray.getConstView();
   auto outputView = outputArray.getView();
   const auto result = pivotPartition( inputView, outputView, pivot );
   verifyBuckets( copyToHost( outputArray ), result, pivot, std::less<>{} );
   verifyPermutation( copyToHost( outputArray ), input );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";

   // in-place overload with a mutable view
   ArrayType dataArray( input );
   auto dataView = dataArray.getView();
   const auto inPlaceResult = pivotPartition( dataView, pivot );
   verifyBuckets( copyToHost( dataArray ), inPlaceResult, pivot, std::less<>{} );
   verifyPermutation( copyToHost( dataArray ), input );
}

TYPED_TEST( PivotPartitionTest, pivotAliasingElement )
{
   using ArrayType = typename TestFixture::ArrayType;
   using DeviceType = typename ArrayType::DeviceType;
   using ValueType = typename ArrayType::ValueType;

   // an element of a device array cannot be aliased from the host code,
   // so the aliasing scenario is tested only on the host devices
   if constexpr( std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential > ) {
      const std::vector< ValueType > input{ 1, 9, 5, 2, 8, 5, 0, 7, 5, 3 };
      const ValueType pivotValue = 5;

      ArrayType dataArray( input );
      // the pivot refers to an element of the partitioned array itself
      const ValueType& pivot = dataArray[ 2 ];

      const auto result = pivotPartition( dataArray, pivot );
      verifyBuckets( copyToHost( dataArray ), result, pivotValue, std::less<>{} );
      verifyPermutation( copyToHost( dataArray ), input );
   }
   else {
      GTEST_SKIP() << "the pivot cannot alias an element of a device array from the host code";
   }
}

TYPED_TEST( PivotPartitionTest, sizeMismatchThrows )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   const ArrayType inputArray( std::vector< ValueType >{ 1, 2, 3, 4, 5 } );
   const ValueType pivot = 3;

   ArrayType smallerOutput( static_cast< IndexType >( 4 ) );
   EXPECT_THROW( (void) pivotPartition( inputArray, smallerOutput, pivot ), std::invalid_argument );

   ArrayType biggerOutput( static_cast< IndexType >( 6 ) );
   EXPECT_THROW( (void) pivotPartition( inputArray, biggerOutput, pivot ), std::invalid_argument );
}

TYPED_TEST( PivotPartitionTest, payloadPreservation )
{
   using ArrayType = typename TestFixture::ArrayType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using PayloadArray = Array< KeyPayload, DeviceType, IndexType >;

   // many elements share the same comparison key and the pivot carries an
   // id which does not occur in the input
   std::vector< KeyPayload > input;
   input.reserve( 100 );
   for( int i = 0; i < 100; i++ )
      input.push_back( KeyPayload{ i % 5, i } );
   const KeyPayload pivot{ 2, -1 };

   // out-of-place overload
   PayloadArray inputArray( input );
   PayloadArray outputArray( static_cast< IndexType >( input.size() ) );
   const auto result = pivotPartition( inputArray, outputArray, pivot, KeyPayloadLess{} );
   verifyPayloadPartition( copyToHost( outputArray ), input, result, pivot, KeyPayloadLess{} );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";

   // in-place overload
   PayloadArray dataArray( input );
   const auto inPlaceResult = pivotPartition( dataArray, pivot, KeyPayloadLess{} );
   verifyPayloadPartition( copyToHost( dataArray ), input, inPlaceResult, pivot, KeyPayloadLess{} );
}

TYPED_TEST( PivotPartitionTest, sizeMatrix )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   // sizes around the GPU thread, tile and block boundaries
   const std::vector< IndexType > sizes{ 0, 1, 2, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 8191, 8192, 8193 };

   for( const IndexType size : sizes ) {
      // deterministic input with many duplicates around the pivot
      std::vector< ValueType > input( static_cast< std::size_t >( size ) );
      std::mt19937 rng( static_cast< std::uint32_t >( size ) + 1 );
      for( std::size_t i = 0; i < input.size(); i++ )
         input[ i ] = static_cast< ValueType >( rng() % 10 );

      verifyDirectedCase< ArrayType >( input, static_cast< ValueType >( 5 ) );
   }
}

TYPED_TEST( PivotPartitionTest, multiBlockRandomized )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   // randomized multi-block inputs, repeated with different seeds;
   // the sizes cross the tile boundaries and exceed one block
   for( int rep = 0; rep < 20; rep++ ) {
      std::mt19937 rng( rep );
      const IndexType size = 2049 + static_cast< IndexType >( rng() % 1000 );
      std::vector< ValueType > input( static_cast< std::size_t >( size ) );
      for( std::size_t i = 0; i < input.size(); i++ )
         input[ i ] = static_cast< ValueType >( rng() % 13 );

      verifyDirectedCase< ArrayType >( input, static_cast< ValueType >( 6 ) );
   }
}

TYPED_TEST( PivotPartitionTest, largeValueForcesDirectScatter )
{
   using ArrayType = typename TestFixture::ArrayType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using LargeArray = Array< LargeValue, DeviceType, IndexType >;

   // many elements share the same comparison key and the pivot carries an
   // id which does not occur in the input; the size of the value type makes
   // the GPU implementation skip the shared-memory staging
   const int size = 5000;
   std::vector< LargeValue > input;
   input.reserve( size );
   for( int i = 0; i < size; i++ )
      input.push_back( makeLargeValue( i % 5, i ) );
   const LargeValue pivot = makeLargeValue( 2, -1 );

   // out-of-place overload
   LargeArray inputArray( input );
   LargeArray outputArray( static_cast< IndexType >( size ) );
   const auto result = pivotPartition( inputArray, outputArray, pivot, LargeValueLess{} );
   verifyPayloadPartition( copyToHost( outputArray ), input, result, pivot, LargeValueLess{} );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";

   // in-place overload
   LargeArray dataArray( input );
   const auto inPlaceResult = pivotPartition( dataArray, pivot, LargeValueLess{} );
   verifyPayloadPartition( copyToHost( dataArray ), input, inPlaceResult, pivot, LargeValueLess{} );
}

TYPED_TEST( PivotPartitionTest, duplicatesAtTileBoundaries )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   // a GPU tile holds 1024 elements: put duplicates of the pivot exactly at
   // the first and the last element of several tiles, where an off-by-one in
   // the per-tile counting or scanning would push them into a wrong bucket
   for( const IndexType size : { IndexType{ 1024 }, IndexType{ 2048 }, IndexType{ 4096 }, IndexType{ 4097 } } ) {
      std::vector< ValueType > input( static_cast< std::size_t >( size ) );
      for( std::size_t i = 0; i < input.size(); i++ )
         input[ i ] = static_cast< ValueType >( i % 3 == 0 ? 6 : i % 7 );

      input[ 0 ] = 5;
      input[ input.size() - 1 ] = 5;
      for( std::size_t i = 1023; i < input.size(); i += 1024 ) {
         input[ i ] = 5;
         if( i + 1 < input.size() )
            input[ i + 1 ] = 5;
      }

      verifyDirectedCase< ArrayType >( input, static_cast< ValueType >( 5 ) );
   }
}

TYPED_TEST( PivotPartitionTest, mediumValueUsesSharedStaging )
{
   using ArrayType = typename TestFixture::ArrayType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using MediumArray = Array< MediumValue, DeviceType, IndexType >;

   // 32-byte values fit the shared-memory budget of one GPU tile, so the GPU
   // implementation stages the elements in the shared memory; the padding
   // must survive the copies through the shared buffer
   const int size = 5000;
   std::vector< MediumValue > input;
   input.reserve( size );
   for( int i = 0; i < size; i++ )
      input.push_back( makeMediumValue( i % 5, i ) );
   const MediumValue pivot = makeMediumValue( 2, -1 );

   const auto verifyPadding = []( const std::vector< MediumValue >& output )
   {
      for( const MediumValue& element : output ) {
         const MediumValue expected = makeMediumValue( element.key, element.id );
         for( int i = 0; i < 24; i++ )
            ASSERT_EQ( element.padding[ i ], expected.padding[ i ] ) << "the padding of the element " << element.id
               << " was corrupted";
      }
   };

   // out-of-place overload
   MediumArray inputArray( input );
   MediumArray outputArray( static_cast< IndexType >( size ) );
   const auto result = pivotPartition( inputArray, outputArray, pivot, MediumValueLess{} );
   const auto outputCopy = copyToHost( outputArray );
   verifyPayloadPartition( outputCopy, input, result, pivot, MediumValueLess{} );
   verifyPadding( outputCopy );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";

   // in-place overload
   MediumArray dataArray( input );
   const auto inPlaceResult = pivotPartition( dataArray, pivot, MediumValueLess{} );
   const auto dataCopy = copyToHost( dataArray );
   verifyPayloadPartition( dataCopy, input, inPlaceResult, pivot, MediumValueLess{} );
   verifyPadding( dataCopy );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
// This test is very slow - run it only when assertions are disabled
   #ifdef NDEBUG
TYPED_TEST( PivotPartitionTest, near32BitSize )
{
   using ArrayType = typename TestFixture::ArrayType;
   using IndexType = typename ArrayType::IndexType;

   if constexpr( sizeof( IndexType ) != sizeof( std::size_t ) )
      GTEST_SKIP() << "the test is instantiated once, for the 64-bit index type";

   // the size crosses the 32-bit boundary, where a 32-bit tile arithmetic or
   // a narrow count type would wrap
   const std::size_t size = ( 1UL << 32 ) + 1024;
   if( TNL::UnitTests::isRunningInCi() && ! TNL::UnitTests::hasEnoughDeviceMemory( size, 1 ) )
      GTEST_SKIP() << "requires about " << 2 * size / ( 1UL << 30 ) << " GiB of free device memory";

   Array< std::uint8_t, Devices::GPU, std::size_t > data( size );
   data.setValue( 0 );
   data.setElement( 0, 1 );
   data.setElement( 1023, 1 );
   data.setElement( 1UL << 32, 2 );
   data.setElement( ( 1UL << 32 ) + 1023, 2 );

   const auto result = pivotPartition( data, std::uint8_t{ 1 } );

   // 2 elements are equivalent to the pivot and 2 are greater than it
   EXPECT_EQ( result.lessEnd, static_cast< std::size_t >( size - 4 ) );
   EXPECT_EQ( result.greaterBegin, static_cast< std::size_t >( size - 2 ) );
   EXPECT_EQ( data.getElement( 0 ), 0 );
   EXPECT_EQ( data.getElement( result.lessEnd - 1 ), 0 );
   EXPECT_EQ( data.getElement( result.lessEnd ), 1 );
   EXPECT_EQ( data.getElement( result.greaterBegin - 1 ), 1 );
   EXPECT_EQ( data.getElement( result.greaterBegin ), 2 );
   EXPECT_EQ( data.getElement( size - 1 ), 2 );
}
   #endif
#endif

#include "../main.h"
