#pragma once

#include <limits>

#include <gtest/gtest.h>

#include <TNL/Arithmetics/experimental/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/reduce.h>
#include "../CustomScalar.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics::experimental;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::detail;

// test fixture for typed tests
template< typename Array >
class ReduceTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ReduceTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Array< CustomScalar< int >, Devices::Sequential, int >,
   Array< int, Devices::Sequential, int >,
   Array< long, Devices::Sequential, int >,
   Array< double, Devices::Sequential, int >,
   //Array< Quad< float >,  Devices::Sequential, int >,
   //Array< Quad< double >, Devices::Sequential, int >,
   Array< CustomScalar< int >, Devices::Sequential, long >,
   Array< int, Devices::Sequential, long >,
   Array< long, Devices::Sequential, long >,
   Array< double, Devices::Sequential, long >,
   //Array< Quad< float >,  Devices::Sequential, long >,
   //Array< Quad< double >, Devices::Sequential, long >,

   Array< CustomScalar< int >, Devices::Host, int >,
   Array< int, Devices::Host, int >,
   Array< long, Devices::Host, int >,
   Array< double, Devices::Host, int >,
   //Array< Quad< float >,  Devices::Host, int >,
   //Array< Quad< double >, Devices::Host, int >,
   Array< CustomScalar< int >, Devices::Host, long >,
   Array< int, Devices::Host, long >,
   Array< long, Devices::Host, long >,
   Array< double, Devices::Host, long >
//Array< Quad< float >,  Devices::Host, long >,
//Array< Quad< double >, Devices::Host, long >
#elif defined( __CUDACC__ )
   // CustomScalar is non-trivially-copyable, so it exercises the shared-memory reduction path
   Array< CustomScalar< int >, Devices::Cuda, int >,
   Array< int, Devices::Cuda, int >,
   Array< long, Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   //Array< Quad< float >,  Devices::Cuda, int >,
   //Array< Quad< double >, Devices::Cuda, int >,
   // CustomScalar is non-trivially-copyable, so it exercises the shared-memory reduction path
   Array< CustomScalar< int >, Devices::Cuda, long >,
   Array< int, Devices::Cuda, long >,
   Array< long, Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >
//Array< Quad< float >,  Devices::Cuda, long >,
//Array< Quad< double >, Devices::Cuda, long >
#elif defined( __HIP__ )
   // CustomScalar is non-trivially-copyable, so it exercises the shared-memory reduction path
   Array< CustomScalar< int >, Devices::Hip, int >,
   Array< int, Devices::Hip, int >,
   Array< long, Devices::Hip, int >,
   Array< double, Devices::Hip, int >,
   //Array< Quad< float >,  Devices::Hip, int >,
   //Array< Quad< double >, Devices::Hip, int >,
   // CustomScalar is non-trivially-copyable, so it exercises the shared-memory reduction path
   Array< CustomScalar< int >, Devices::Hip, long >,
   Array< int, Devices::Hip, long >,
   Array< long, Devices::Hip, long >,
   Array< double, Devices::Hip, long >
//Array< Quad< float >,  Devices::Hip, long >,
//Array< Quad< double >, Devices::Hip, long >
#endif
   >;

TYPED_TEST_SUITE( ReduceTest, ArrayTypes );

template< typename Array >
class ReduceShortTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// Arrays with short IndexType have a much smaller range than the general tests above.
using ShortArrayTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Array< double, Devices::Sequential, short >,
   Array< CustomScalar< int >, Devices::Sequential, short >,
   Array< double, Devices::Host, short >,
   Array< CustomScalar< int >, Devices::Host, short >
#elif defined( __CUDACC__ )
   Array< double, Devices::Cuda, short >,
   Array< CustomScalar< int >, Devices::Cuda, short >
#elif defined( __HIP__ )
   Array< double, Devices::Hip, short >,
   Array< CustomScalar< int >, Devices::Hip, short >
#endif
   >;

TYPED_TEST_SUITE( ReduceShortTest, ShortArrayTypes );

template< typename Array >
void
iota( Array& array, typename Array::ValueType start = 0 )
{
   array.forAllElements(
      [ start ] __cuda_callable__( typename Array::IndexType idx, typename Array::ValueType& value )
      {
         value = idx + start;
      } );
}

template< typename Array >
void
mod( Array& array, typename Array::IndexType mod = 0 )
{
   array.forAllElements(
      [ mod ] __cuda_callable__( typename Array::IndexType idx, typename Array::ValueType& value )
      {
         value = idx % mod;
      } );
}

TYPED_TEST( ReduceTest, sum )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      a.setValue( 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Plus{} );
      EXPECT_EQ( res, size );

      res = reduce( a, TNL::Plus{} );
      EXPECT_EQ( res, size );
   }

   const int size = 9377;
   a.setSize( size );
   iota( a );
   auto res = reduce( a, TNL::Plus{} );
   EXPECT_EQ( res, ( size * ( size - 1 ) ) / 2 );
}

TYPED_TEST( ReduceTest, product )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   a.setSize( 10 );
   a.setValue( 2 );

   int result = 1;
   for( int size = 0; size < a.getSize(); size++ ) {
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Multiplies{} );
      EXPECT_EQ( res, result );
      result *= 2;
   }
}

TYPED_TEST( ReduceTest, min )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Min{} );
      EXPECT_EQ( res, 1 );
   }
}

TYPED_TEST( ReduceTest, max )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Max{} );
      EXPECT_EQ( res, size );
   }
}

TYPED_TEST( ReduceTest, minWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::MinWithArg{} );
      EXPECT_EQ( res.first, 1 );
      EXPECT_EQ( res.second, 0 );
   }
}

TYPED_TEST( ReduceTest, maxWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, size );
      EXPECT_EQ( res.second, size - 1 );
   }
}

TYPED_TEST( ReduceShortTest, sum )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   ArrayType a;
   for( const int sizeInt : { 1, 255, 256, 257, 1000, 32512, 32513, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      a.setSize( size );
      a.setValue( 1 );

      auto res = reduce< typename ArrayType::DeviceType >( (Index) 0, size, a.getConstView(), TNL::Plus{} );
      EXPECT_EQ( res, size );
   }
}

TYPED_TEST( ReduceShortTest, maxWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   ArrayType a;
   for( const int sizeInt : { 1, 255, 256, 257, 1000, 32512, 32513, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( (Index) 0, size, a.getConstView(), TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, size );
      EXPECT_EQ( res.second, size - 1 );
   }
}

TYPED_TEST( ReduceShortTest, minWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   ArrayType a;
   for( const int sizeInt : { 1, 255, 256, 257, 1000, 32512, 32513, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( (Index) 0, size, a.getConstView(), TNL::MinWithArg{} );
      EXPECT_EQ( res.first, 1 );
      EXPECT_EQ( res.second, 0 );
   }
}

TYPED_TEST( ReduceShortTest, maxWithArgNonZeroBegin )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   ArrayType a;
   for( const int sizeInt : { 1000, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      const Index begin = 10;
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, size );
      EXPECT_EQ( res.second, size - 1 );
   }
}

TYPED_TEST( ReduceShortTest, sumNonZeroBegin )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   ArrayType a;
   for( const int sizeInt : { 1000, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      const Index begin = 10;
      a.setSize( size );
      a.setValue( 1 );

      auto res = reduce< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::Plus{} );
      EXPECT_EQ( res, size - begin );
   }
}

TYPED_TEST( ReduceShortTest, beginNearMax )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   const Index size = 32767;
   const Index begin = 32000;
   ArrayType a;
   a.setSize( size );
   a.setValue( 1 );

   auto sum = reduce< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::Plus{} );
   EXPECT_EQ( sum, size - begin );

   iota( a, 1 );
   auto res = reduceWithArgument< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::MaxWithArg{} );
   EXPECT_EQ( res.first, size );
   EXPECT_EQ( res.second, size - 1 );
}

TYPED_TEST( ReduceShortTest, maxWithArgAllIdentityNonZeroBegin )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   using Value = typename ArrayType::ValueType;
   ArrayType a;
   for( const int sizeInt : { 1000, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      const Index begin = 10;
      a.setSize( size );
      a.setValue( std::numeric_limits< Value >::lowest() );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, std::numeric_limits< Value >::lowest() );
      EXPECT_EQ( res.second, begin );
   }
}

TYPED_TEST( ReduceShortTest, minWithArgAllIdentityNonZeroBegin )
{
   using ArrayType = typename TestFixture::ArrayType;
   using Index = typename ArrayType::IndexType;
   using Value = typename ArrayType::ValueType;
   ArrayType a;
   for( const int sizeInt : { 1000, 32767 } ) {
      const Index size = static_cast< Index >( sizeInt );
      const Index begin = 10;
      a.setSize( size );
      a.setValue( std::numeric_limits< Value >::max() );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( begin, size, a.getConstView(), TNL::MinWithArg{} );
      EXPECT_EQ( res.first, std::numeric_limits< Value >::max() );
      EXPECT_EQ( res.second, begin );
   }
}

template< typename Device >
void
testAnyWithArgShort()
{
   using Index = short;
   auto first_true = [] __cuda_callable__( Index i )
   {
      return i == 17;
   };
   auto res = reduceWithArgument< Device >( (Index) 10, (Index) 100, first_true, TNL::AnyWithArg{} );
   EXPECT_TRUE( res.first );
   EXPECT_EQ( res.second, (Index) 17 );

   auto all_false = [] __cuda_callable__( Index )
   {
      return false;
   };
   res = reduceWithArgument< Device >( (Index) 10, (Index) 100, all_false, TNL::AnyWithArg{} );
   EXPECT_FALSE( res.first );
   EXPECT_EQ( res.second, (Index) 10 );
}

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
TEST( ReduceShortAnyTest, anyWithArg )
{
   testAnyWithArgShort< Devices::Sequential >();
   testAnyWithArgShort< Devices::Host >();
}
#elif defined( __CUDACC__ )
TEST( ReduceShortAnyTest, anyWithArg )
{
   testAnyWithArgShort< Devices::Cuda >();
}
#elif defined( __HIP__ )
TEST( ReduceShortAnyTest, anyWithArg )
{
   testAnyWithArgShort< Devices::Hip >();
}
#endif

TYPED_TEST( ReduceTest, logicalAnd )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );

      mod( a, 2 );
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalAnd{} );
      EXPECT_EQ( res, false );

      a.setValue( 1 );
      res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalAnd{} );
      EXPECT_EQ( res, true );
   }
}

TYPED_TEST( ReduceTest, logicalOr )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );

      if( size == 1 )
         a.setValue( 1 );
      else
         mod( a, 2 );
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalOr{} );
      EXPECT_EQ( res, true );

      a.setValue( 0 );
      res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalOr{} );
      EXPECT_EQ( res, false );
   }
}

// bitwise AND (&) is not defined for floating-point types
template< typename ArrayType >
std::enable_if_t< std::is_integral_v< typename ArrayType::ValueType > >
test_bitAnd( ArrayType& a )
{
   for( int size = 1; size <= 100000; size *= 10 ) {
      a.setSize( size );
      a.forAllElements(
         [] __cuda_callable__( typename ArrayType::IndexType idx, typename ArrayType::ValueType& value )
         {
            value = 1 | ( 1 << ( idx % 8 ) );
         } );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::BitAnd{} );
      EXPECT_EQ( res, 1 );
   }
}

template< typename ArrayType >
std::enable_if_t< ! std::is_integral_v< typename ArrayType::ValueType > >
test_bitAnd( ArrayType& a )
{}

TYPED_TEST( ReduceTest, bitAnd )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   test_bitAnd( a );
}

// bitwise OR (|) is not defined for floating-point types
template< typename ArrayType >
std::enable_if_t< std::is_integral_v< typename ArrayType::ValueType > >
test_bitOr( ArrayType& a )
{
   for( int size = 10; size <= 100000; size *= 10 ) {
      a.setSize( size );
      a.forAllElements(
         [] __cuda_callable__( typename ArrayType::IndexType idx, typename ArrayType::ValueType& value )
         {
            value = 1 << ( idx % 8 );
         } );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::BitOr{} );
      EXPECT_EQ( res, 255 );
   }
}

template< typename ArrayType >
std::enable_if_t< ! std::is_integral_v< typename ArrayType::ValueType > >
test_bitOr( ArrayType& a )
{}

TYPED_TEST( ReduceTest, bitOr )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   test_bitOr( a );
}

template< typename ArrayType >
void
test_ReduceTest_powersOfTwoSizes()
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;

   // For GPU devices, size will be max 2^32 for 64-bit IndexType and 2^16 for 32-bit IndexType.
   // For host and sequential, it is just 2^16 to keep the test reasonably fast.
   constexpr int max_exp = std::is_same_v< DeviceType, Devices::GPU > ? sizeof( IndexType ) * 4 : sizeof( int ) * 4;
   for( int exp = max_exp; exp >= 0; exp -= 4 ) {
      const IndexType size = static_cast< IndexType >( 1 ) << exp;

      // Use a lambda rather than Array to avoid allocation
      auto fetch = [ size ] __cuda_callable__( IndexType i ) -> ValueType
      {
         if( i == size - 1 )
            return 1;
         else
            return 0;
      };

      auto res = reduce< DeviceType, IndexType >( 0, size, fetch, TNL::Plus{} );
      EXPECT_EQ( res, 1 ) << "failed for size 2^" << exp << " = " << size;
   }
}

TYPED_TEST( ReduceTest, powersOfTwoSizes )
{
   test_ReduceTest_powersOfTwoSizes< typename TestFixture::ArrayType >();
}

#include "../main.h"
