#pragma once

#include <type_traits>

#include <TNL/Algorithms/MultipleReductions.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

// larger than minGPUReductionDataSize and large enough to require multiple GPU blocks
constexpr int multipleReductionsTestSize = 5000;

// the values from -50 to 50 in a pseudorandom order
template< typename Vector >
typename Vector::template Self< typename Vector::RealType, Devices::Sequential >
getMultipleReductionsTestValues( int shift )
{
   typename Vector::template Self< typename Vector::RealType, Devices::Sequential > values( multipleReductionsTestSize );
   for( int i = 0; i < multipleReductionsTestSize; i++ )
      values[ i ] = typename Vector::RealType( ( i * 37 + shift ) % 101 - 50 );
   return values;
}

// several different reductions of one vector, including a reduction with an argument
template< typename Vector >
void
test_oneVector()
{
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using ArgumentType = ValueWithArgument< RealType, IndexType >;

   const auto hostValues = getMultipleReductionsTestValues< Vector >( 0 );
   Vector values;
   values = hostValues;
   const auto v = values.getConstView();

   auto fetch = [ = ] __cuda_callable__( IndexType i )
   {
      const RealType x = v[ i ];
      return makeReductionTuple( x, x, x, x * x, ArgumentType{ x, i } );
   };
   const auto [ sum, maximum, minimum, squares, maxWithArg ] = reduce< DeviceType >(
      0, multipleReductionsTestSize, fetch, MultipleReductions< Plus, Max, Min, Plus, WithArgument< MaxWithArg > >{} );

   RealType expectedSum = 0;
   RealType expectedSquares = 0;
   for( int i = 0; i < multipleReductionsTestSize; i++ ) {
      expectedSum += hostValues[ i ];
      expectedSquares += hostValues[ i ] * hostValues[ i ];
   }
   EXPECT_EQ( sum, expectedSum );
   EXPECT_EQ( maximum, 50 );
   EXPECT_EQ( minimum, -50 );
   EXPECT_EQ( squares, expectedSquares );
   EXPECT_EQ( maxWithArg.value, 50 );
   // i * 37 % 101 == 100 for the first time at i = 30
   EXPECT_EQ( maxWithArg.argument, 30 );
}

// different reductions of several vectors of the same size
template< typename Vector >
void
test_severalVectors()
{
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;

   const auto hostU = getMultipleReductionsTestValues< Vector >( 0 );
   const auto hostW = getMultipleReductionsTestValues< Vector >( 13 );
   Vector u;
   Vector w;
   u = hostU;
   w = hostW;
   const auto uView = u.getConstView();
   const auto wView = w.getConstView();

   auto fetch = [ = ] __cuda_callable__( IndexType i )
   {
      return makeReductionTuple( uView[ i ] * wView[ i ], TNL::abs( wView[ i ] ), uView[ i ] > 49, wView[ i ] > 50 );
   };
   const auto result =
      reduce< DeviceType >( 0, multipleReductionsTestSize, fetch, MultipleReductions< Plus, Max, LogicalOr, LogicalOr >{} );

   RealType expectedDot = 0;
   for( int i = 0; i < multipleReductionsTestSize; i++ )
      expectedDot += hostU[ i ] * hostW[ i ];
   EXPECT_EQ( result.template get< 0 >(), expectedDot );
   EXPECT_EQ( result.template get< 1 >(), 50 );
   EXPECT_TRUE( result.template get< 2 >() );
   EXPECT_FALSE( result.template get< 3 >() );
}

// several different reductions in every segment
template< typename Vector >
void
test_segments()
{
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;

   // the segments of the lengths 0, 1, ..., segmentsCount - 1 cover the values
   constexpr int segmentsCount = 100;
   Containers::Vector< IndexType, DeviceType, IndexType > sizes( segmentsCount );
   sizes.forAllElements(
      [] __cuda_callable__( IndexType i, IndexType& value )
      {
         value = i;
      } );
   const Segments::CSR< DeviceType, IndexType > segments( sizes );
   const auto hostValues = getMultipleReductionsTestValues< Vector >( 0 );
   Vector values;
   values = hostValues;
   const auto v = values.getConstView();

   Vector sums( segmentsCount );
   Vector maxima( segmentsCount );
   Containers::Vector< IndexType, DeviceType, IndexType > counts( segmentsCount );
   auto sumsView = sums.getView();
   auto maximaView = maxima.getView();
   auto countsView = counts.getView();
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx )
   {
      return makeReductionTuple( v[ globalIdx ], v[ globalIdx ], IndexType( v[ globalIdx ] > 0 ) );
   };
   using ResultType = ReductionTuple< RealType, RealType, IndexType >;
   auto storer = [ = ] __cuda_callable__( IndexType segmentIdx, const ResultType& result ) mutable
   {
      sumsView[ segmentIdx ] = result.template get< 0 >();
      maximaView[ segmentIdx ] = result.template get< 1 >();
      countsView[ segmentIdx ] = result.template get< 2 >();
   };
   Segments::reduceAllSegments( segments, fetch, MultipleReductions< Plus, Max, Plus >{}, storer );

   int offset = 0;
   for( int segment = 0; segment < segmentsCount; segment++ ) {
      RealType sum = 0;
      RealType maximum = Max::getIdentity< RealType >();
      IndexType count = 0;
      for( int i = offset; i < offset + segment; i++ ) {
         sum += hostValues[ i ];
         maximum = TNL::max( maximum, hostValues[ i ] );
         count += hostValues[ i ] > 0;
      }
      offset += segment;
      EXPECT_EQ( sums.getElement( segment ), sum ) << "segment = " << segment;
      EXPECT_EQ( maxima.getElement( segment ), maximum ) << "segment = " << segment;
      EXPECT_EQ( counts.getElement( segment ), count ) << "segment = " << segment;
   }
}

// test fixture for typed tests
template< typename Vector >
class MultipleReductionsTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
};

// types for which MultipleReductionsTest is instantiated
using VectorTypes = ::testing::Types<  //
   Vector< int, Devices::Host >,
   Vector< float, Devices::Host >,
   Vector< int, Devices::Sequential >,
   Vector< float, Devices::Sequential >
#if defined( __CUDACC__ )
   ,
   Vector< int, Devices::Cuda >,
   Vector< float, Devices::Cuda >
#endif
#if defined( __HIP__ )
   ,
   Vector< int, Devices::Hip >,
   Vector< float, Devices::Hip >
#endif
   >;

TYPED_TEST_SUITE( MultipleReductionsTest, VectorTypes );

TYPED_TEST( MultipleReductionsTest, oneVector )
{
   test_oneVector< typename TestFixture::VectorType >();
}

TYPED_TEST( MultipleReductionsTest, severalVectors )
{
   test_severalVectors< typename TestFixture::VectorType >();
}

TYPED_TEST( MultipleReductionsTest, segments )
{
   using VectorType = typename TestFixture::VectorType;
   // the segments are not implemented for the sequential device
   if constexpr( ! std::is_same_v< typename VectorType::DeviceType, Devices::Sequential > )
      test_segments< VectorType >();
}

#include "../main.h"
