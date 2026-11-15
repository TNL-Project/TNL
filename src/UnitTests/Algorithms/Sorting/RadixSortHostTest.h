#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/RadixSort.h>

#include "RadixSortTestUtils.h"

using namespace TNL;
using namespace TNL::Algorithms;

/**
 * \brief Pair of the value and device types of a typed test.
 */
template< typename Value, typename Device >
struct RadixSortHostParams
{
   using ValueType = Value;
   using DeviceType = Device;
};

template< typename Params >
class RadixSortHostTest : public ::testing::Test
{
protected:
   using ValueType = typename Params::ValueType;
   using DeviceType = typename Params::DeviceType;
   using ArrayType = Containers::Array< ValueType, DeviceType, int >;
   using WorkspaceType = Sorting::RadixSort::Workspace< ValueType, DeviceType, int >;

   /**
    * \brief Sorts `vec` by the radix sort (optionally with a workspace) and
    * compares the result with `std::sort`.
    */
   static void
   checkSort( std::vector< ValueType > vec, const bool descending = false, WorkspaceType* workspace = nullptr )
   {
      ArrayType arr( vec );
      if( descending ) {
         if( workspace != nullptr )
            Sorting::RadixSort::sort( arr, *workspace, std::greater{} );
         else
            Sorting::RadixSort::sort( arr, std::greater{} );
         std::sort( vec.begin(), vec.end(), std::greater<>{} );
      }
      else {
         if( workspace != nullptr )
            Sorting::RadixSort::sort( arr, *workspace );
         else
            Sorting::RadixSort::sort( arr );
         std::sort( vec.begin(), vec.end() );
      }
      const ArrayType expected( vec );
      EXPECT_EQ( arr, expected ) << "failed for size " << vec.size() << ( descending ? " (descending)" : " (ascending)" );
   }
};

#define RADIX_SORT_HOST_PARAMS( Value ) \
   RadixSortHostParams< Value, Devices::Sequential >, RadixSortHostParams< Value, Devices::Host >

using RadixSortHostTypes = ::testing::Types<
   RADIX_SORT_HOST_PARAMS( std::int8_t ),
   RADIX_SORT_HOST_PARAMS( std::uint8_t ),
   RADIX_SORT_HOST_PARAMS( std::int16_t ),
   RADIX_SORT_HOST_PARAMS( std::uint16_t ),
   RADIX_SORT_HOST_PARAMS( int ),
   RADIX_SORT_HOST_PARAMS( unsigned int ),
   RADIX_SORT_HOST_PARAMS( std::int64_t ),
   RADIX_SORT_HOST_PARAMS( std::uint64_t ),
   RADIX_SORT_HOST_PARAMS( float ),
   RADIX_SORT_HOST_PARAMS( double ) >;

#undef RADIX_SORT_HOST_PARAMS

TYPED_TEST_SUITE( RadixSortHostTest, RadixSortHostTypes );

// Edge case tests

TYPED_TEST( RadixSortHostTest, emptyArray )
{
   typename TestFixture::ArrayType arr;
   Sorting::RadixSort::sort( arr );
   EXPECT_EQ( arr.getSize(), 0 );
}

TYPED_TEST( RadixSortHostTest, fewElements )
{
   using Value = typename TestFixture::ValueType;
   this->checkSort( { Value( 42 ) } );
   this->checkSort( { Value( 1 ), Value( 2 ) } );
   this->checkSort( { Value( 2 ), Value( 1 ) } );
   this->checkSort( { Value( 5 ), Value( 5 ) } );
   this->checkSort( { Value( 2 ), Value( 1 ) }, true );
}

TYPED_TEST( RadixSortHostTest, alreadySortedAndReverseSorted )
{
   using Value = typename TestFixture::ValueType;
   for( const int size : { 64, 65, 1024, 300000 } ) {
      std::vector< Value > vec( size );
      // the values wrap around for the narrow types, which is fine
      for( int i = 0; i < size; i++ )
         vec[ i ] = static_cast< Value >( i );
      std::sort( vec.begin(), vec.end() );
      this->checkSort( vec );
      std::reverse( vec.begin(), vec.end() );
      this->checkSort( vec );
   }
}

TYPED_TEST( RadixSortHostTest, allIdentical )
{
   using Value = typename TestFixture::ValueType;
   // all passes are skipped
   for( const int size : { 64, 1024, 300000 } )
      this->checkSort( std::vector< Value >( size, static_cast< Value >( 42 ) ) );
}

// Random data tests

TYPED_TEST( RadixSortHostTest, random_smallArrays )
{
   using Value = typename TestFixture::ValueType;
   std::mt19937 rng( 2006 );
   // covers both sides of the insertion sort limit
   for( int size = 1; size <= 300; size++ )
      this->checkSort( makeRandomVector< Value >( size, rng() ) );
   for( int i = 0; i < 50; i++ ) {
      const int size = rng() % ( 1 << 12 ) + 1;
      this->checkSort( makeRandomVector< Value >( size, rng() ) );
   }
}

TYPED_TEST( RadixSortHostTest, random_bigArrays )
{
   using Value = typename TestFixture::ValueType;
   std::mt19937 rng( 304 );
   // sizes below, around and well above the minimal chunk of the parallel sort
   for( const int size : { ( 1 << 16 ) - 1, 1 << 17, ( 1 << 17 ) + 1, 999999, ( 1 << 21 ) + 7 } )
      this->checkSort( makeRandomVector< Value >( size, rng() ) );
}

TYPED_TEST( RadixSortHostTest, descendingOrder )
{
   using Value = typename TestFixture::ValueType;
   std::mt19937 rng( 42 );
   for( const int size : { 50, 1000, 1 << 18 } )
      this->checkSort( makeRandomVector< Value >( size, rng() ), true );
}

// Duplicate-heavy tests (many equal digits per pass)

TYPED_TEST( RadixSortHostTest, duplicates_smallAlphabet )
{
   using Value = typename TestFixture::ValueType;
   for( std::uint32_t seed = 0; seed < 3; seed++ ) {
      for( const int size : { 1000, 1 << 18 } ) {
         std::vector< Value > vec( size );
         std::mt19937 rng( seed );
         // only the lowest digit varies, so all other passes are skipped
         // and the result ends up in the auxiliary array
         for( auto& value : vec )
            value = static_cast< Value >( static_cast< int >( rng() % 4 ) );
         this->checkSort( vec );
         this->checkSort( vec, true );
      }
   }
}

TYPED_TEST( RadixSortHostTest, negativeValues )
{
   using Value = typename TestFixture::ValueType;
   const int size = 1 << 18;
   std::vector< Value > vec( size );
   for( int i = 0; i < size; i++ )
      vec[ i ] = static_cast< Value >( -( i % 1000 ) );
   this->checkSort( vec );
}

TYPED_TEST( RadixSortHostTest, extremeValues )
{
   using Value = typename TestFixture::ValueType;
   using Limits = std::numeric_limits< Value >;
   std::vector< Value > vec = makeRandomVector< Value >( 1 << 17, 777 );
   for( int i = 0; i < 1000; i++ ) {
      vec[ 7 * i ] = Limits::max();
      vec[ 7 * i + 1 ] = Limits::lowest();
      vec[ 7 * i + 2 ] = Value( 0 );
      if constexpr( Limits::has_infinity ) {
         vec[ 7 * i + 3 ] = Limits::infinity();
         vec[ 7 * i + 4 ] = -Limits::infinity();
      }
   }
   this->checkSort( vec );
   this->checkSort( vec, true );
}

// Workspace tests

TYPED_TEST( RadixSortHostTest, workspace_growingAndShrinkingSizes )
{
   using Value = typename TestFixture::ValueType;
   typename TestFixture::WorkspaceType workspace;
   std::mt19937 rng( 7001 );
   for( const int size : { 2, 17, 1024, ( 1 << 18 ) + 3, 100, 5000, 1 << 17 } ) {
      this->checkSort( makeRandomVector< Value >( size, rng() ), false, &workspace );
      this->checkSort( makeRandomVector< Value >( size, rng() ), true, &workspace );
   }
   // the buffers are never shrunk
   EXPECT_GE( workspace.aux.getSize(), ( 1 << 18 ) + 3 );
}

TYPED_TEST( RadixSortHostTest, workspace_smallArrayIsNotAllocated )
{
   using Value = typename TestFixture::ValueType;
   typename TestFixture::WorkspaceType workspace;
   this->checkSort( makeRandomVector< Value >( 64, 3 ), false, &workspace );
   EXPECT_EQ( workspace.aux.getSize(), 0 );
}

// Tests of the parallel sort with various thread counts

TEST( RadixSortHostThreadsTest, oddThreadCounts )
{
   const int originalThreads = Devices::Host::getMaxThreadsCount();
   std::mt19937 rng( 1234 );
   for( const int threads : { 1, 2, 3, 5, 7, 16 } ) {
      Devices::Host::setMaxThreadsCount( threads );
      for( const int size : { ( 1 << 16 ) * 3 - 1, ( 1 << 20 ) + 13 } ) {
         std::vector< int > vec = makeRandomVector< int >( size, rng() );
         Containers::Array< int, Devices::Host > arr( vec );
         Sorting::RadixSort::sort( arr );
         std::sort( vec.begin(), vec.end() );
         const Containers::Array< int, Devices::Host > expected( vec );
         EXPECT_EQ( arr, expected ) << "failed for " << threads << " threads and size " << size;

         std::vector< double > vecD = makeRandomVector< double >( size, rng() );
         Containers::Array< double, Devices::Host > arrD( vecD );
         Sorting::RadixSort::sort( arrD, std::greater{} );
         std::sort( vecD.begin(), vecD.end(), std::greater<>{} );
         const Containers::Array< double, Devices::Host > expectedD( vecD );
         EXPECT_EQ( arrD, expectedD ) << "failed for " << threads << " threads and size " << size;
      }
   }
   Devices::Host::setMaxThreadsCount( originalThreads );
}

// Arrays larger than the limit of the non-temporal stores of the scatter

template< typename Value, typename Device >
void
checkHugeArray( const int size, const std::uint32_t seed )
{
   std::vector< Value > vec = makeRandomVector< Value >( size, seed );
   Containers::Array< Value, Device > arr( vec );
   Sorting::RadixSort::sort( arr );
   std::sort( vec.begin(), vec.end() );
   const Containers::Array< Value, Device > expected( vec );
   EXPECT_EQ( arr, expected ) << "failed for size " << size;
}

TEST( RadixSortHostHugeArrayTest, nontemporalStores )
{
   // 2^23 + 5 values of 4 and 8 bytes are above the 32 MiB limit
   checkHugeArray< int, Devices::Host >( ( 1 << 23 ) + 5, 31 );
   checkHugeArray< double, Devices::Host >( ( 1 << 23 ) + 5, 32 );
   checkHugeArray< std::uint64_t, Devices::Sequential >( ( 1 << 23 ) + 5, 33 );
   checkHugeArray< unsigned int, Devices::Sequential >( ( 1 << 23 ) + 5, 34 );
}

// Floating-point special values

template< typename Device >
void
checkSignedZeros()
{
   // the radix sort orders -0.0 before +0.0 (unlike std::sort, which treats
   // them as equal) on all devices
   std::vector< float > vec( 1 << 17 );
   std::mt19937 rng( 99 );
   for( auto& value : vec )
      value = ( rng() % 2 == 0 ) ? -0.0f : 0.0f;
   vec[ 0 ] = 1.0f;
   vec[ 1 ] = -1.0f;
   Containers::Array< float, Device > arr( vec );
   Sorting::RadixSort::sort( arr );
   EXPECT_TRUE( Algorithms::isAscending( arr ) );
   const int negativeZeros = static_cast< int >( std::count_if(
      vec.begin(),
      vec.end(),
      []( float x )
      {
         return x == 0.0f && std::signbit( x );
      } ) );
   EXPECT_EQ( arr.getElement( 0 ), -1.0f );
   for( int i = 1; i <= negativeZeros; i++ )
      ASSERT_TRUE( arr.getElement( i ) == 0.0f && std::signbit( arr.getElement( i ) ) ) << "i = " << i;
   for( int i = negativeZeros + 1; i < arr.getSize() - 1; i++ )
      ASSERT_TRUE( arr.getElement( i ) == 0.0f && ! std::signbit( arr.getElement( i ) ) ) << "i = " << i;
   EXPECT_EQ( arr.getElement( arr.getSize() - 1 ), 1.0f );
}

TEST( RadixSortHostSpecialValuesTest, signedZeros )
{
   checkSignedZeros< Devices::Sequential >();
   checkSignedZeros< Devices::Host >();
}

// Integration with free sorting functions

TEST( RadixSortHostFreeFunctionTest, ascendingAndDescending )
{
   std::vector< int > vec = makeRandomVector< int >( 1 << 18, 123 );

   Containers::Array< int, Devices::Host > arr( vec );
   Algorithms::ascendingSort( arr, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isAscending( arr ) );
   Algorithms::descendingSort( arr, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isDescending( arr ) );

   Containers::Array< int, Devices::Sequential > arrSeq( vec );
   Algorithms::ascendingSort( arrSeq, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isAscending( arrSeq ) );
}

// Index type tests

template< typename Device, typename Index >
void
checkIndexType()
{
   for( const Index size : { Index( 10000 ), Index( 1 << 18 ) } ) {
      std::vector< int > vec( size );
      std::iota( vec.begin(), vec.end(), 0 );
      std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );
      Containers::Array< int, Device, Index > arr( vec );
      Sorting::RadixSort::sort( arr );
      for( Index i = 0; i < size; i++ )
         ASSERT_EQ( arr[ i ], static_cast< int >( i ) );
   }
}

TEST( RadixSortHostIndexTest, indexTypes )
{
   checkIndexType< Devices::Sequential, std::int64_t >();
   checkIndexType< Devices::Host, std::int64_t >();
   checkIndexType< Devices::Sequential, std::size_t >();
   checkIndexType< Devices::Host, std::size_t >();
}

#include "../../main.h"
