#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

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
      1000000 ) );

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
   BitonicSort::sort( view );
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
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

   // This test is very slow - run it only when assertions are disabled
   #ifdef NDEBUG
TYPED_TEST( SorterTest, large_index_size2To32 )
{
   const std::size_t size = 1UL << 32;
   TNL::Containers::Array< std::uint8_t, TNL::Devices::GPU, std::size_t > arr( size );
   arr.setValue( 0 );
   arr.setElement( 0, 1 );
   arr.setElement( 1, 2 );
   arr.setElement( size - 1, 3 );

   auto view = arr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size 2^32 = " << size;
}
   #endif

// =============================================================================
// BitonicSort-specific tests (fetch-and-swap interface)
// =============================================================================

namespace {
template< typename ValueType >
void
fetchAndSwapSorter( TNL::Containers::ArrayView< ValueType, TNL::Devices::GPU > view )
{
   using Index = typename decltype( view )::IndexType;
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

#endif

#include "../../main.h"
