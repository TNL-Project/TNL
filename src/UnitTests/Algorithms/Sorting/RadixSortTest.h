#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/RadixSort.h>
#include <TNL/Backend.h>

#if defined( __CUDACC__ ) || defined( __HIP__ )

using namespace TNL;
using namespace TNL::Algorithms;

/**
 * \brief Generates one random value of the given type.
 */
template< typename Value >
Value
randomValue( std::mt19937& rng )
{
   if constexpr( std::is_floating_point_v< Value > ) {
      std::uniform_real_distribution< Value > dist( -1000.0, 1000.0 );
      return dist( rng );
   }
   else if constexpr( sizeof( Value ) == 1 ) {
      const int value = static_cast< int >( rng() % 256 ) - ( std::is_signed_v< Value > ? 128 : 0 );
      return static_cast< Value >( value );
   }
   else if constexpr( std::is_signed_v< Value > ) {
      using Distribution = std::conditional_t<
         sizeof( Value ) == 4,
         std::uniform_int_distribution< int >,
         std::uniform_int_distribution< std::int64_t > >;
      using Result = typename Distribution::result_type;
      Distribution dist( std::numeric_limits< Result >::min() / 2, std::numeric_limits< Result >::max() / 2 );
      return static_cast< Value >( dist( rng ) );
   }
   else {
      using Distribution = std::conditional_t<
         sizeof( Value ) == 4,
         std::uniform_int_distribution< unsigned int >,
         std::uniform_int_distribution< std::uint64_t > >;
      using Result = typename Distribution::result_type;
      Distribution dist( 0, std::numeric_limits< Result >::max() / 2 );
      return static_cast< Value >( dist( rng ) );
   }
}

/**
 * \brief Generates a vector of random values of the given type.
 */
template< typename Value >
std::vector< Value >
makeRandomVector( std::size_t size, std::uint32_t seed )
{
   std::vector< Value > vec( size );
   std::mt19937 rng( seed );
   for( auto& value : vec )
      value = randomValue< Value >( rng );
   return vec;
}

// Typed tests over supported value types

using ValueTypes = ::testing::
   Types< std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, int, unsigned int, std::int64_t, std::uint64_t, float, double >;

template< typename Value >
class RadixSortTest : public ::testing::Test
{};

TYPED_TEST_SUITE( RadixSortTest, ValueTypes );

// Edge case tests

TYPED_TEST( RadixSortTest, emptyArray )
{
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr;
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_EQ( view.getSize(), 0 );
}

TYPED_TEST( RadixSortTest, singleElement )
{
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr{ 42 };
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, twoElements_sorted )
{
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr{ 1, 2 };
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, twoElements_unsorted )
{
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr{ 2, 1 };
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, twoElements_equal )
{
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr{ 5, 5 };
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, alreadySorted )
{
   const int size = 1024;
   std::vector< TypeParam > vec( size );
   std::iota( vec.begin(), vec.end(), static_cast< TypeParam >( 0 ) );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, reverseSorted )
{
   const int size = 1024;
   std::vector< TypeParam > vec( size );
   std::iota( vec.rbegin(), vec.rend(), static_cast< TypeParam >( 0 ) );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, allIdentical )
{
   const int size = 1024;
   std::vector< TypeParam > vec( size, static_cast< TypeParam >( 42 ) );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// Random data tests

TYPED_TEST( RadixSortTest, random_smallArrays )
{
   std::mt19937 rng( 2006 );
   for( int i = 0; i < 100; i++ ) {
      const int size = rng() % ( 1 << 10 ) + 1;
      std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, rng() );

      TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      Sorting::RadixSort::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
   }
}

TYPED_TEST( RadixSortTest, random_bigArrays )
{
   std::mt19937 rng( 304 );
   for( int i = 0; i < 20; i++ ) {
      const int size = ( rng() % ( 1 << 18 ) ) + 1;
      std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, rng() );

      TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      Sorting::RadixSort::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
   }
}

// No lost element tests (exact comparison against std::sort)

TYPED_TEST( RadixSortTest, noLostElement_midSizedArray )
{
   const int size = 1 << 15;
   std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, 91503 );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );

   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );
}

TYPED_TEST( RadixSortTest, noLostElement_bigSizedArray )
{
   const int size = 1 << 22;
   std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, 15611 );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );

   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );
}

// Duplicate-heavy tests (many equal digits per pass)

TYPED_TEST( RadixSortTest, duplicates_smallAlphabet )
{
   for( std::uint32_t seed = 0; seed < 5; seed++ ) {
      const int size = 1 << 16;
      std::vector< TypeParam > vec( size );
      std::mt19937 rng( seed );
      for( auto& value : vec )
         value = static_cast< TypeParam >( static_cast< int >( rng() % 4 ) );

      TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      Sorting::RadixSort::sort( view );

      std::sort( vec.begin(), vec.end() );
      TNL::Containers::Array< TypeParam, TNL::Devices::GPU > sortedArr( vec );
      EXPECT_TRUE( view == sortedArr.getView() ) << "failed for seed " << seed;
   }
}

TYPED_TEST( RadixSortTest, duplicates_zeroEntropy )
{
   const int size = 1 << 16;
   std::vector< TypeParam > vec( size, static_cast< TypeParam >( 7 ) );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// Negative values tests

TYPED_TEST( RadixSortTest, negativeValues )
{
   const int size = 1 << 15;
   std::vector< TypeParam > vec( size );
   for( int i = 0; i < size; i++ )
      vec[ i ] = static_cast< TypeParam >( -( i % 1000 ) );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// Descending order tests

TYPED_TEST( RadixSortTest, descendingOrder )
{
   const int size = 1 << 14;
   std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, 42 );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view, std::greater{} );
   EXPECT_TRUE( Algorithms::isDescending( view ) );
}

// Workspace tests (reuse of the temporary buffers across sorts)

TYPED_TEST( RadixSortTest, workspace_growingSizes )
{
   using Workspace = Sorting::RadixSort::Workspace< TypeParam, TNL::Devices::GPU, int >;
   Workspace workspace;
   std::mt19937 rng( 7001 );
   // the sizes grow, so the buffers of the workspace are reallocated on
   // each sort; the last sort is too large to allocate per call, so an
   // exhausted buffer pool would corrupt the result
   for( const int size : { 2, 17, 1024, ( 1 << 16 ) + 3 } ) {
      std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, rng() );

      TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      Sorting::RadixSort::sort( view, workspace );
      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
   }
}

TYPED_TEST( RadixSortTest, workspace_neverShrinks )
{
   using Workspace = Sorting::RadixSort::Workspace< TypeParam, TNL::Devices::GPU, int >;
   constexpr int passes = static_cast< int >( sizeof( TypeParam ) );
   Workspace workspace;

   // grow the buffers by a large sort
   const int bigSize = 1 << 16;
   std::vector< TypeParam > vec = makeRandomVector< TypeParam >( bigSize, 517 );
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   Sorting::RadixSort::sort( arr, workspace );
   const int grownAuxSize = workspace.aux.getSize();
   EXPECT_GE( grownAuxSize, bigSize );
   EXPECT_EQ( workspace.digitBase.getSize(), passes * 256 );
   EXPECT_EQ( workspace.passFlags.getSize(), 2 * passes + 1 );

   // a smaller sort keeps the grown buffers
   vec = makeRandomVector< TypeParam >( 100, 518 );
   arr = vec;
   Sorting::RadixSort::sort( arr, workspace );
   EXPECT_TRUE( Algorithms::isAscending( arr ) );
   EXPECT_EQ( workspace.aux.getSize(), grownAuxSize );
}

TYPED_TEST( RadixSortTest, workspace_zeroEntropyAndEqualElements )
{
   using Workspace = Sorting::RadixSort::Workspace< TypeParam, TNL::Devices::GPU, int >;
   Workspace workspace;
   const int size = 1 << 16;

   // zero-entropy input: the histogram of every pass owns a single non-zero
   // bucket, so all passes are skipped and no sweep runs
   std::vector< TypeParam > vec( size, static_cast< TypeParam >( 7 ) );
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view, workspace );
   EXPECT_TRUE( Algorithms::isAscending( view ) );

   // a few distinct values on a small alphabet: most passes execute after
   // the skipped ones, reusing the buffers of the skipped sort
   std::mt19937 rng( 910 );
   for( auto& value : vec )
      value = static_cast< TypeParam >( static_cast< int >( rng() % 4 ) );
   arr = vec;
   Sorting::RadixSort::sort( view, workspace );
   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );

   // a random sort must not observe any state of the two orders above
   vec = makeRandomVector< TypeParam >( size, 911 );
   arr = vec;
   Sorting::RadixSort::sort( view, workspace );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( RadixSortTest, workspace_descendingOrder )
{
   using Workspace = Sorting::RadixSort::Workspace< TypeParam, TNL::Devices::GPU, int >;
   Workspace workspace;

   // ascending and descending sorts alternate on one workspace
   const int size = 1 << 14;
   std::vector< TypeParam > vec = makeRandomVector< TypeParam >( size, 4242 );

   TNL::Containers::Array< TypeParam, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view, workspace );
   EXPECT_TRUE( Algorithms::isAscending( view ) );

   Sorting::RadixSort::sort( view, workspace, std::greater{} );
   EXPECT_TRUE( Algorithms::isDescending( view ) );
}

// Single-tile threshold tests (both sides of the single-tile dispatch)

/**
 * \brief Checks the sort of a `size`-element random vector of the given
 * value type with both the plain and the workspace overload, ascending and
 * descending.
 */
template< typename Value >
void
checkSortAroundSingleTileThreshold( int size )
{
   using Workspace = Sorting::RadixSort::Workspace< Value, TNL::Devices::GPU, int >;
   std::vector< Value > vec = makeRandomVector< Value >( size, 4210 + size );
   std::vector< Value > expected( vec );
   TNL::Containers::Array< Value, TNL::Devices::GPU, int > arr( vec );
   auto view = arr.getView();
   TNL::Containers::Array< Value, TNL::Devices::GPU, int > expectedArr;

   // plain overloads
   Sorting::RadixSort::sort( view );
   std::sort( expected.begin(), expected.end() );
   expectedArr = expected;
   EXPECT_TRUE( view == expectedArr.getView() ) << "ascending (plain) failed for size " << size;

   arr = vec;
   Sorting::RadixSort::sort( view, std::greater{} );
   std::sort( expected.begin(), expected.end(), std::greater<>{} );
   expectedArr = expected;
   EXPECT_TRUE( view == expectedArr.getView() ) << "descending (plain) failed for size " << size;

   // workspace overloads (the workspace must stay empty on the single-tile
   // path and grow only above the threshold)
   Sorting::RadixSort::Workspace< Value, TNL::Devices::GPU, int > workspace;
   arr = vec;
   Sorting::RadixSort::sort( view, workspace );
   std::sort( expected.begin(), expected.end() );
   expectedArr = expected;
   EXPECT_TRUE( view == expectedArr.getView() ) << "ascending (workspace) failed for size " << size;
   EXPECT_EQ( workspace.aux.getSize(), size <= 2048 ? 0 : size );

   arr = vec;
   Sorting::RadixSort::sort( view, workspace, std::greater{} );
   std::sort( expected.begin(), expected.end(), std::greater<>{} );
   expectedArr = expected;
   EXPECT_TRUE( view == expectedArr.getView() ) << "descending (workspace) failed for size " << size;
}

TEST( RadixSortSingleTileTest, thresholdBoundary )
{
   // the single-tile fast path handles at most one tile (2048 elements), so
   // test one element below, at, and one element above the threshold
   for( const int size : { 2047, 2048, 2049 } ) {
      checkSortAroundSingleTileThreshold< int >( size );
      checkSortAroundSingleTileThreshold< float >( size );
   }
}

// Integration with free sorting functions

TEST( RadixSortFreeFunctionTest, ascendingAndDescending )
{
   const int size = 1 << 10;
   std::vector< int > vec = makeRandomVector< int >( size, 123 );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   Algorithms::ascendingSort( arr, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isAscending( arr ) );

   Algorithms::descendingSort( arr, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isDescending( arr ) );
}

// The single-pass variant assigns tiles to blocks via a work counter, so
// that each block may claim multiple tiles. The size is chosen just above
// the number of blocks launched by the sort (32 per multiprocessor), which
// pins the tile-reuse path of the binning kernel.
TEST( RadixSortFreeFunctionTest, multipleTilesPerBlock )
{
   const int multiprocessors = Backend::getDeviceMultiprocessors( Backend::getDevice() );
   const int size = ( 32 * multiprocessors + 3 ) * 2048;
   std::vector< int > vec = makeRandomVector< int >( size, 456 );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   Algorithms::ascendingSort( arr, Sorting::RadixSort{} );
   EXPECT_TRUE( Algorithms::isAscending( arr ) ) << "failed for size " << size;
}

// 64-bit index tests

TEST( RadixSortLargeIndexTest, index_int64 )
{
   using Index = std::int64_t;
   const Index size = 10000;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU, Index > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( RadixSortLargeIndexTest, index_size_t )
{
   using Index = std::size_t;
   const Index size = 10000;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU, Index > arr( vec );
   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

   // This test is very slow - run it only when assertions are disabled
   #ifdef NDEBUG
TEST( RadixSortLargeIndexTest, index_size2To32 )
{
      #if defined( __HIP__ )
   if( std::getenv( "GITLAB_CI" ) != nullptr && Backend::getDeviceMultiprocessors( Backend::getDevice() ) <= 2 )
      GTEST_SKIP() << "Skipping on HIP CI with <= 2 CUs (takes ~10 min on 2-CU iGPU)";
      #endif

   const std::size_t size = 1UL << 32;
   TNL::Containers::Array< std::uint8_t, TNL::Devices::GPU, std::size_t > arr( size );
   arr.setValue( 0 );
   arr.setElement( 0, 1 );
   arr.setElement( 1, 2 );
   arr.setElement( size - 1, 3 );

   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size 2^32 = " << size;
}

TEST( RadixSortLargeIndexTest, index_int32CloseToMax )
{
   // exercises the overflow-prone index arithmetic with a signed index type
   // and an array size close to the maximum of int
   const int size = std::numeric_limits< int >::max() - 100;
   TNL::Containers::Array< std::uint8_t, TNL::Devices::GPU, int > arr( size );
   arr.setValue( 0 );
   arr.setElement( 0, 1 );
   arr.setElement( 1, 2 );
   arr.setElement( size - 1, 3 );

   auto view = arr.getView();
   Sorting::RadixSort::sort( view );
   // isAscending runs the generic GPU reduction, which is also index-overflow
   // sensitive, over the whole (nearly INT_MAX-sized) array
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}
   #endif

#endif

#include "../../main.h"
