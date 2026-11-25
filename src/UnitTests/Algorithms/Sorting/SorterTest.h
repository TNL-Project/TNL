#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>
#include <cstdint>

#include <TNL/Backend/DeviceInfo.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/experimental/Quicksort.h>
#include <TNL/Algorithms/Sorting/CUBMergeSort.h>

#if defined( __CUDACC__ ) || defined( __HIP__ )

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

using Sorters = ::testing::Types<
   BitonicSort,
   experimental::Quicksort
   #if defined( __CUDACC__ )
   ,
   CUBMergeSort
   #endif
   >;

template< typename Sorter >
class SorterTest : public ::testing::Test
{};

TYPED_TEST_SUITE( SorterTest, Sorters );

// =============================================================================
// Edge case tests
// =============================================================================

TYPED_TEST( SorterTest, emptyArray )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr;
   auto view = arr.getView();
   EXPECT_EQ( view.getSize(), 0 );
   TypeParam::sort( view );
   EXPECT_EQ( view.getSize(), 0 );
}

TYPED_TEST( SorterTest, singleElement )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr{ 42 };
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, twoElements_sorted )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr{ 1, 2 };
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, twoElements_unsorted )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr{ 2, 1 };
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, twoElements_equal )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr{ 5, 5 };
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, alreadySorted )
{
   const int size = 1024;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, reverseSorted )
{
   const int size = 1024;
   std::vector< int > vec( size );
   std::iota( vec.rbegin(), vec.rend(), 0 );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, allIdentical )
{
   const int size = 1024;
   std::vector< int > vec( size, 42 );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Power-of-2 and non-power-of-2 sizes
// =============================================================================

class SorterPowerOfTwoSizes : public ::testing::TestWithParam< int >
{};

TEST_P( SorterPowerOfTwoSizes, bitonicSort )
{
   const int size = GetParam();
   std::vector< int > vec( size );
   std::mt19937 rng( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}

INSTANTIATE_TEST_SUITE_P(
   sorterTests,
   SorterPowerOfTwoSizes,
   ::testing::Values(
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096,
      8192,
      16384,
      32768,
      65536,
      131072,
      262144,
      524288,
      1048576,
      2097152,
      4194304,
      8388608,
      16777216 ) );

class SorterNonPowerOfTwoSizes : public ::testing::TestWithParam< int >
{};

TEST_P( SorterNonPowerOfTwoSizes, bitonicSort )
{
   const int size = GetParam();
   std::vector< int > vec( size );
   std::mt19937 rng( size + 12345 );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}

INSTANTIATE_TEST_SUITE_P(
   sorterTests,
   SorterNonPowerOfTwoSizes,
   ::testing::Values(
      3,
      5,
      6,
      7,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      17,
      31,
      33,
      63,
      65,
      127,
      129,
      1000,
      1001,
      1023,
      1025,
      2047,
      2049,
      4095,
      4097,
      10000,
      100000,
      1000000,
      // Above 2048 * 8 * 150 = 2 457 600, so that the shared-memory merge
      // kernels that handle several pairs per thread are covered together with
      // a partial last tile on GPUs with up to 150 multiprocessors (B300 has
      // 148); below that threshold the small tile keeps one pair per thread.
      3000001,
      16777215 ) );

TEST( BitonicSortNonPowerOfTwo, doubleValues )
{
   const int size = 3000001;

   std::mt19937 rng( 1234 );
   std::vector< double > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return static_cast< double >( rng() );
      } );

   TNL::Containers::Array< double, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}

// =============================================================================
// Permutation tests
// =============================================================================

TYPED_TEST( SorterTest, permutations_size2_to_7 )
{
   for( int size = 2; size <= 7; size++ ) {
      std::vector< int > orig( size );
      std::iota( orig.begin(), orig.end(), 0 );

      do {
         TNL::Containers::Array< int, TNL::Devices::GPU > arr( orig );
         auto view = arr.getView();
         TypeParam::sort( view );
         EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
      } while( std::next_permutation( orig.begin(), orig.end() ) );
   }
}

TYPED_TEST( SorterTest, permutations_sampled_size8 )
{
   const int size = 8;
   const int stride = 151;
   int count = 0;

   std::vector< int > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      if( ( count++ ) % stride != 0 )
         continue;

      TNL::Containers::Array< int, TNL::Devices::GPU > arr( orig );
      auto view = arr.getView();
      TypeParam::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

// =============================================================================
// Basic functionality tests
// =============================================================================

TYPED_TEST( SorterTest, selectedSize_size15 )
{
   TNL::Containers::Array< int, TNL::Devices::GPU > arr{ 5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9 };
   auto view = arr.getView();
   EXPECT_EQ( 15, view.getSize() );
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, multiblock_32768_decreasingNegative )
{
   std::vector< int > vec( 1 << 15 );
   for( size_t i = 0; i < vec.size(); i++ )
      vec[ i ] = -i;

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Random data tests
// =============================================================================

TYPED_TEST( SorterTest, random_smallArrays )
{
   std::mt19937 rng( 2006 );
   for( int i = 0; i < 100; i++ ) {
      int size = rng() % ( 1 << 10 ) + 1;
      std::vector< int > vec( size );
      std::generate(
         vec.begin(),
         vec.end(),
         [ & ]()
         {
            return rng();
         } );

      TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      TypeParam::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   }
}

TYPED_TEST( SorterTest, random_bigArrays )
{
   std::mt19937 rng( 304 );
   for( int i = 0; i < 20; i++ ) {
      int size = ( rng() % ( 1 << 18 ) ) + 1;
      std::vector< int > vec( size );
      std::generate(
         vec.begin(),
         vec.end(),
         [ & ]()
         {
            return rng();
         } );

      TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
      auto view = arr.getView();
      TypeParam::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   }
}

// =============================================================================
// No lost element tests
// =============================================================================

TYPED_TEST( SorterTest, noLostElement_smallArray )
{
   std::mt19937 rng( 9151 );

   int size = 1 << 7;
   std::vector< int > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );

   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< int, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );
}

TYPED_TEST( SorterTest, noLostElement_midSizedArray )
{
   std::mt19937 rng( 91503 );

   int size = 1 << 15;
   std::vector< int > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );

   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< int, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );
}

TYPED_TEST( SorterTest, noLostElement_bigSizedArray )
{
   std::mt19937 rng( 15611 );

   int size = 1 << 22;
   std::vector< int > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );

   std::sort( vec.begin(), vec.end() );
   TNL::Containers::Array< int, TNL::Devices::GPU > sortedArr( vec );
   EXPECT_TRUE( view == sortedArr.getView() );
}

// =============================================================================
// Value type tests
// =============================================================================

TYPED_TEST( SorterTest, valueType_int32 )
{
   std::vector< int > vec( 1000 );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, valueType_int64 )
{
   std::vector< std::int64_t > vec( 1000 );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< std::int64_t, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, valueType_uint8 )
{
   std::vector< std::uint8_t > vec( 256 );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< std::uint8_t, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, valueType_float32 )
{
   std::mt19937 rng( 8451 );
   std::uniform_real_distribution< float > dist( -1000.0f, 1000.0f );
   int size = 1 << 16;
   std::vector< float > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return dist( rng );
      } );

   TNL::Containers::Array< float, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, valueType_float64 )
{
   std::mt19937 rng( 8451 );
   std::uniform_real_distribution< double > dist( -1000.0, 1000.0 );
   int size = 1 << 16;
   std::vector< double > vec( size );
   std::generate(
      vec.begin(),
      vec.end(),
      [ & ]()
      {
         return dist( rng );
      } );

   TNL::Containers::Array< double, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Custom struct tests
// =============================================================================

struct TestStruct3D
{
   double x, y, z;
   __cuda_callable__
   TestStruct3D()
   : x( 0 )
   {}
   __cuda_callable__
   TestStruct3D( int val )
   : x( val )
   {}
   __cuda_callable__
   bool
   operator<( const TestStruct3D& other ) const
   {
      return x < other.x;
   }
   __cuda_callable__
   TestStruct3D&
   operator=( const TestStruct3D& other )
   {
      x = other.x;
      return *this;
   }
};

std::ostream&
operator<<( std::ostream& out, const TestStruct3D& data )
{
   return out << data.x;
}

TYPED_TEST( SorterTest, struct_3D_points )
{
   std::mt19937 rng( 46151 );
   int size = 1 << 18;
   std::vector< TestStruct3D > vec( size );
   for( auto& x : vec )
      x = TestStruct3D( rng() );

   TNL::Containers::Array< TestStruct3D, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

struct TestStruct64B
{
   uint8_t m_data[ 64 ];
   __cuda_callable__
   TestStruct64B()
   {
      m_data[ 0 ] = 0;
   }
   __cuda_callable__
   TestStruct64B( int val )
   {
      m_data[ 0 ] = val;
   }
   __cuda_callable__
   bool
   operator<( const TestStruct64B& other ) const
   {
      return m_data[ 0 ] < other.m_data[ 0 ];
   }
   __cuda_callable__
   TestStruct64B&
   operator=( const TestStruct64B& other )
   {
      m_data[ 0 ] = other.m_data[ 0 ];
      return *this;
   }
};

std::ostream&
operator<<( std::ostream& out, const TestStruct64B& data )
{
   return out << (unsigned) data.m_data[ 0 ];
}

TYPED_TEST( SorterTest, struct_64b )
{
   std::mt19937 rng( 96 );
   int size = 1 << 18;
   std::vector< TestStruct64B > vec( size );
   for( auto& x : vec )
      x = TestStruct64B( rng() % 512 );

   TNL::Containers::Array< TestStruct64B, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// 64-bit index tests
// =============================================================================

TYPED_TEST( SorterTest, large_index_int64 )
{
   using Index = std::int64_t;
   const Index size = 10000;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU, Index > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TYPED_TEST( SorterTest, large_index_size_t )
{
   using Index = std::size_t;
   const Index size = 10000;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU, Index > arr( vec );
   auto view = arr.getView();
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

   // This test is very slow - run it only when assertions are disabled
   #ifdef NDEBUG
TYPED_TEST( SorterTest, large_index_size2To32 )
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
   TypeParam::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size 2^32 = " << size;
}
   #endif

// =============================================================================
// BitonicSort-specific tests (fetch-and-swap interface)
// =============================================================================

namespace {
template< typename ValueType, typename Index >
void
fetchAndSwapSorter( TNL::Containers::ArrayView< ValueType, TNL::Devices::GPU, Index > view )
{
   auto Cmp = [ = ] __cuda_callable__( Index i, Index j )
   {
      return view[ i ] < view[ j ];
   };
   auto Swap = [ = ] __cuda_callable__( Index i, Index j ) mutable
   {
      TNL::swap( view[ i ], view[ j ] );
   };
   Sorting::detail::bitonicSort( Index{ 0 }, view.getSize(), Cmp, Swap );
}

void
fetchAndSwap_sortMiddle( TNL::Containers::ArrayView< int, TNL::Devices::GPU > view, int from, int to )
{
   auto Cmp = [ = ] __cuda_callable__( int i, int j )
   {
      return view[ i ] < view[ j ];
   };
   auto Swap = [ = ] __cuda_callable__( int i, int j ) mutable
   {
      TNL::swap( view[ i ], view[ j ] );
   };
   Sorting::detail::bitonicSort( from, to, Cmp, Swap );
}

void
fetchAndSwap_trackPayload(
   TNL::Containers::ArrayView< int, TNL::Devices::GPU > keys,
   TNL::Containers::ArrayView< int, TNL::Devices::GPU > payload )
{
   using Index = typename decltype( keys )::IndexType;
   auto Cmp = [ = ] __cuda_callable__( Index i, Index j )
   {
      return keys[ i ] < keys[ j ];
   };
   auto Swap = [ = ] __cuda_callable__( Index i, Index j ) mutable
   {
      TNL::swap( keys[ i ], keys[ j ] );
      TNL::swap( payload[ i ], payload[ j ] );
   };
   Sorting::detail::bitonicSort( Index{ 0 }, keys.getSize(), Cmp, Swap );
}

}  // namespace

TEST( BitonicSortFetchAndSwap, smallArray )
{
   int size = 9;
   const int stride = 227;
   int count = 0;

   std::vector< int > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      if( ( count++ ) % stride != 0 )
         continue;

      TNL::Containers::Array< int, TNL::Devices::GPU > arr( orig );
      auto view = arr.getView();
      fetchAndSwapSorter( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

TEST( BitonicSortFetchAndSwap, typeDouble )
{
   int size = 5;
   std::vector< double > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      TNL::Containers::Array< double, TNL::Devices::GPU > arr( orig );
      auto view = arr.getView();
      fetchAndSwapSorter( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

TEST( BitonicSortFetchAndSwap, subrange )
{
   std::vector< int > orig{ 5, 9, 4, 54, 21, 6, 7, 9, 0, 9, 42, 4 };
   TNL::Containers::Array< int, TNL::Devices::GPU > arr( orig );
   auto view = arr.getView();
   int from = 3;
   int to = 8;

   fetchAndSwap_sortMiddle( view, from, to );
   EXPECT_TRUE( Algorithms::isAscending( view.getView( 3, 8 ) ) );

   for( std::size_t i = 0; i < orig.size(); i++ ) {
      if( i < static_cast< std::size_t >( from ) || i >= static_cast< std::size_t >( to ) )
         EXPECT_EQ( view.getElement( i ), orig[ i ] );
   }
}

TEST( BitonicSortFetchAndSwap, largeArray )
{
   const int size = 1 << 20;
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::shuffle( vec.begin(), vec.end(), std::mt19937( 12345 ) );

   TNL::Containers::Array< int, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   fetchAndSwapSorter( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( BitonicSortFetchAndSwap, trackPermutation )
{
   const int size = 64;
   std::mt19937 rng( 12345 );

   // unique keys make the tracked position of every key unambiguous
   std::vector< int > keys( size );
   std::iota( keys.begin(), keys.end(), 0 );
   std::shuffle( keys.begin(), keys.end(), rng );
   std::vector< int > payload( size );
   std::iota( payload.begin(), payload.end(), 0 );

   TNL::Containers::Array< int, TNL::Devices::GPU > keysArr( keys );
   TNL::Containers::Array< int, TNL::Devices::GPU > payloadArr( payload );

   auto keysView = keysArr.getView();
   auto payloadView = payloadArr.getView();
   fetchAndSwap_trackPayload( keysView, payloadView );

   EXPECT_TRUE( Algorithms::isAscending( keysView ) );

   // the payload has to stay a permutation and to keep the original position of each key
   std::vector< bool > seen( size, false );
   for( int i = 0; i < size; i++ ) {
      const int originalIndex = payloadView.getElement( i );
      ASSERT_GE( originalIndex, 0 );
      ASSERT_LT( originalIndex, size );
      EXPECT_FALSE( seen[ originalIndex ] );
      seen[ originalIndex ] = true;
      EXPECT_EQ( keys[ originalIndex ], keysView.getElement( i ) );
   }
   EXPECT_EQ( std::count( seen.begin(), seen.end(), true ), size );
}

// The padded sequence length has to fit the type used for indexing inside the kernels,
// which is widened to at least 32 bits even when the range is given by a narrow signed type
TEST( BitonicSortFetchAndSwap, narrowSignedIndex )
{
   using Index = std::int16_t;
   // 2^14 + 1 elements need a padded length of 2^15, which does not fit in int16_t
   // but does fit in the 32-bit index type used by the kernels
   const Index size = ( 1 << 14 ) + 1;

   std::mt19937 rng( 12345 );
   std::vector< int > vec( size );
   for( auto& x : vec )
      x = rng() % 1000;

   TNL::Containers::Array< int, TNL::Devices::GPU, Index > arr( vec );
   fetchAndSwapSorter( arr.getView() );
   EXPECT_TRUE( Algorithms::isAscending( arr.getView() ) );
}

// Value types large enough to reduce the block size or to force the global memory fallback
template< int Padding >
struct LargeValue
{
   std::uint8_t m_data[ Padding ]{};
   __cuda_callable__
   LargeValue()
   {
      m_data[ 0 ] = 0;
   }
   __cuda_callable__
   LargeValue( int val )
   {
      m_data[ 0 ] = static_cast< std::uint8_t >( val );
   }
   __cuda_callable__
   bool
   operator<( const LargeValue& other ) const
   {
      return m_data[ 0 ] < other.m_data[ 0 ];
   }
   // Assigning the whole padding inside the sorting kernels exceeds the
   // launch resource limit for the 576 B case, so only the key is copied.
   __cuda_callable__
   LargeValue&
   operator=( const LargeValue& other )
   {
      m_data[ 0 ] = other.m_data[ 0 ];
      return *this;
   }
};

template< int Padding >
std::ostream&
operator<<( std::ostream& out, const LargeValue< Padding >& data )
{
   return out << static_cast< unsigned >( data.m_data[ 0 ] );
}

template< int Padding >
void
sortLargeValues( int size, std::mt19937& rng )
{
   std::vector< LargeValue< Padding > > vec( size );
   for( auto& x : vec )
      x = LargeValue< Padding >( rng() % 256 );

   TNL::Containers::Array< LargeValue< Padding >, TNL::Devices::GPU > arr( vec );
   auto view = arr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for " << sizeof( LargeValue< Padding > ) << " B values";
}

TEST( BitonicSortLargeValue, value128B )
{
   std::mt19937 rng( 1 );
   sortLargeValues< 128 >( 4096, rng );
}

TEST( BitonicSortLargeValue, value256B )
{
   std::mt19937 rng( 2 );
   sortLargeValues< 256 >( 4096, rng );
}

TEST( BitonicSortLargeValue, value576B )
{
   std::mt19937 rng( 3 );
   sortLargeValues< 576 >( 4096, rng );
}

// =============================================================================
// Block bitonic sort tests
// =============================================================================

namespace {

// Number of blocks for the block sort tests. It is smaller than the number of
// tiles, so each block sorts several tiles and reuses its shared memory.
constexpr int blockSortGridSize = 2;

// Each block sorts tiles of `src` and writes them to the same tiles of `dst`.
template< typename Value, typename Index >
__global__
void
blockSortSharedKernel(
   TNL::Containers::ArrayView< Value, TNL::Devices::GPU, Index > src,
   TNL::Containers::ArrayView< Value, TNL::Devices::GPU, Index > dst,
   Index tileSize )
{
   extern __shared__ int sharedMem[];
   // 64-bit tile bounds avoid overflow of narrow index types
   const std::int64_t size = src.getSize();
   for( std::int64_t begin = blockIdx.x * std::int64_t( tileSize ); begin < size;
        begin += gridDim.x * std::int64_t( tileSize ) )
   {
      const Index b = static_cast< Index >( begin );
      const Index e = static_cast< Index >( TNL::min( begin + tileSize, size ) );
      Sorting::detail::bitonicSort_Block(
         src.getView( b, e ), dst.getView( b, e ), reinterpret_cast< Value* >( sharedMem ), std::less<>{} );
   }
}

// Each block sorts tiles of `data` in place.
template< typename Value, typename Index >
__global__
void
blockSortInplaceKernel( TNL::Containers::ArrayView< Value, TNL::Devices::GPU, Index > data, Index tileSize )
{
   // 64-bit tile bounds avoid overflow of narrow index types
   const std::int64_t size = data.getSize();
   for( std::int64_t begin = blockIdx.x * std::int64_t( tileSize ); begin < size;
        begin += gridDim.x * std::int64_t( tileSize ) )
   {
      const Index b = static_cast< Index >( begin );
      const Index e = static_cast< Index >( TNL::min( begin + tileSize, size ) );
      Sorting::detail::bitonicSort_Block( data.getView( b, e ), std::greater<>{} );
   }
}

// Sorts the tiles on the host to get the expected result.
template< typename Value, typename Compare >
std::vector< Value >
sortTilesOnHost( std::vector< Value > vec, int tileSize, const Compare& compare )
{
   for( std::size_t begin = 0; begin < vec.size(); begin += tileSize ) {
      const std::size_t end = std::min( begin + tileSize, vec.size() );
      std::sort( vec.begin() + begin, vec.begin() + end, compare );
   }
   return vec;
}

}  // namespace

TEST( BitonicSortBlock, sharedMemory )
{
   std::mt19937 rng( 4 );
   for( int blockSize : { 32, 128, 256 } )
      for( int tileSize : { 1, 2, 7, 64, 300, 1024, 2047 } ) {
         const int size = 5 * tileSize + tileSize / 3;
         std::vector< int > vec( size );
         for( auto& x : vec )
            x = rng() % 1000 - 500;

         TNL::Containers::Array< int, TNL::Devices::GPU > src( vec );
         TNL::Containers::Array< int, TNL::Devices::GPU > dst( size );
         TNL::Backend::LaunchConfiguration config;
         config.blockSize.x = blockSize;
         config.gridSize.x = blockSortGridSize;
         config.dynamicSharedMemorySize = tileSize * sizeof( int );
         constexpr auto kernel = blockSortSharedKernel< int, int >;
         TNL::Backend::launchKernelSync( kernel, config, src.getView(), dst.getView(), tileSize );

         const TNL::Containers::Array< int, TNL::Devices::GPU > expected( sortTilesOnHost( vec, tileSize, std::less<>{} ) );
         EXPECT_EQ( dst, expected ) << "blockSize = " << blockSize << ", tileSize = " << tileSize;
         // the source array must not be modified
         const TNL::Containers::Array< int, TNL::Devices::GPU > original( vec );
         EXPECT_EQ( src, original );
      }
}

TEST( BitonicSortBlock, sharedMemoryInplace )
{
   std::mt19937 rng( 5 );
   const int blockSize = 128;
   const int tileSize = 1000;
   const int size = 3 * tileSize + 17;
   std::vector< double > vec( size );
   for( auto& x : vec )
      x = static_cast< double >( rng() % 10000 ) / 7.0;

   TNL::Containers::Array< double, TNL::Devices::GPU > arr( vec );
   TNL::Backend::LaunchConfiguration config;
   config.blockSize.x = blockSize;
   config.gridSize.x = blockSortGridSize;
   config.dynamicSharedMemorySize = tileSize * sizeof( double );
   constexpr auto kernel = blockSortSharedKernel< double, int >;
   TNL::Backend::launchKernelSync( kernel, config, arr.getView(), arr.getView(), tileSize );

   TNL::Containers::Array< double, TNL::Devices::GPU > expected( sortTilesOnHost( vec, tileSize, std::less<>{} ) );
   EXPECT_EQ( arr, expected );
}

TEST( BitonicSortBlock, globalMemory )
{
   std::mt19937 rng( 6 );
   for( int blockSize : { 32, 128, 256 } )
      for( int tileSize : { 1, 2, 7, 64, 300, 1024, 5000 } ) {
         const int size = 4 * tileSize + tileSize / 2;
         std::vector< std::int64_t > vec( size );
         for( auto& x : vec )
            x = static_cast< std::int64_t >( rng() % 1000 ) - 500;

         TNL::Containers::Array< std::int64_t, TNL::Devices::GPU, long > arr( vec );
         TNL::Backend::LaunchConfiguration config;
         config.blockSize.x = blockSize;
         config.gridSize.x = blockSortGridSize;
         constexpr auto kernel = blockSortInplaceKernel< std::int64_t, long >;
         TNL::Backend::launchKernelSync( kernel, config, arr.getView(), long( tileSize ) );

         TNL::Containers::Array< std::int64_t, TNL::Devices::GPU, long > expected(
            sortTilesOnHost( vec, tileSize, std::greater<>{} ) );
         EXPECT_EQ( arr, expected ) << "blockSize = " << blockSize << ", tileSize = " << tileSize;
      }
}

// The padded tile size 2^15 does not fit into the 16-bit index type.
TEST( BitonicSortBlock, narrowIndex )
{
   std::mt19937 rng( 7 );
   const short tileSize = 20000;
   const short size = 30000;
   std::vector< int > vec( size );
   for( auto& x : vec )
      x = static_cast< int >( rng() % 100000 );

   TNL::Containers::Array< int, TNL::Devices::GPU, short > arr( vec );
   TNL::Backend::LaunchConfiguration config;
   config.blockSize.x = 256;
   config.gridSize.x = blockSortGridSize;
   constexpr auto kernel = blockSortInplaceKernel< int, short >;
   TNL::Backend::launchKernelSync( kernel, config, arr.getView(), tileSize );

   const TNL::Containers::Array< int, TNL::Devices::GPU, short > expected( sortTilesOnHost( vec, tileSize, std::greater<>{} ) );
   EXPECT_EQ( arr, expected );
}

#endif

#include "../../main.h"
