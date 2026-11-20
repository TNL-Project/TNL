// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>    // std::sort
#include <cstddef>      // std::size_t
#include <functional>   // std::greater, std::less
#include <stdexcept>    // std::invalid_argument
#include <type_traits>  // std::is_signed_v
#include <vector>

#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/pivotPartition.h>
#include <TNL/Containers/Array.h>

#include "gtest/gtest.h"

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
   constexpr bool
   operator()( const KeyPayload& lhs, const KeyPayload& rhs ) const
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
// once in the output, and no payload is reconstructed from the pivot
template< typename Index >
void
verifyPayloadPartition(
   const std::vector< KeyPayload >& output,
   const std::vector< KeyPayload >& input,
   const PivotPartitionResult< Index >& result,
   const KeyPayload& pivot )
{
   verifyBuckets( output, result, pivot, KeyPayloadLess{} );

   // every input id must occur exactly once in the output and no element
   // equivalent to the pivot may be reconstructed from the pivot
   std::vector< int > idCounts( input.size(), 0 );
   for( const KeyPayload& element : output ) {
      EXPECT_NE( element.id, pivot.id ) << "an element equivalent to the pivot was reconstructed from the pivot";
      ASSERT_GE( element.id, 0 );
      ASSERT_LT( element.id, static_cast< int >( input.size() ) );
      ++idCounts[ element.id ];
   }
   for( const int count : idCounts ) {
      EXPECT_EQ( count, 1 ) << "each input payload must occur exactly once in the output";
   }

   // the multiset of (key, id) pairs must be unchanged
   const auto byKeyAndId = []( const KeyPayload& lhs, const KeyPayload& rhs )
   {
      return lhs.key != rhs.key ? lhs.key < rhs.key : lhs.id < rhs.id;
   };
   std::vector< KeyPayload > sortedOutput = output;
   std::vector< KeyPayload > sortedInput = input;
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
   Array< int, Devices::Sequential, int >,
   Array< double, Devices::Sequential, long >,
   Array< int, Devices::Host, int >,
   Array< double, Devices::Host, long >,
   Array< std::size_t, Devices::Host, std::size_t > >;

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
   using ValueType = typename ArrayType::ValueType;

   const std::vector< ValueType > input{ 1, 9, 5, 2, 8, 5, 0, 7, 5, 3 };
   const ValueType pivotValue = 5;

   ArrayType dataArray( input );
   // the pivot refers to an element of the partitioned array itself
   const ValueType& pivot = dataArray[ 2 ];

   const auto result = pivotPartition( dataArray, pivot );
   verifyBuckets( copyToHost( dataArray ), result, pivotValue, std::less<>{} );
   verifyPermutation( copyToHost( dataArray ), input );
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
   verifyPayloadPartition( copyToHost( outputArray ), input, result, pivot );
   EXPECT_EQ( copyToHost( inputArray ), input ) << "the input must stay unchanged";

   // in-place overload
   PayloadArray dataArray( input );
   const auto inPlaceResult = pivotPartition( dataArray, pivot, KeyPayloadLess{} );
   verifyPayloadPartition( copyToHost( dataArray ), input, inPlaceResult, pivot );
}

#include "../main.h"
