#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/sort.h>

#if defined( __CUDACC__ ) || defined( __HIP__ )
using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;
using namespace TNL::Algorithms::Sorting::detail;

// =============================================================================
// Edge case tests
// =============================================================================

TEST( edgeCases, emptyArray )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr;
   auto view = cudaArr.getView();
   EXPECT_EQ( view.getSize(), 0 );
   BitonicSort::sort( view );
   EXPECT_EQ( view.getSize(), 0 );
}

TEST( edgeCases, singleElement )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ 42 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_EQ( view.getElement( 0 ), 42 );
}

TEST( edgeCases, twoElements_sorted )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ 1, 2 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( edgeCases, twoElements_unsorted )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ 2, 1 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( edgeCases, twoElements_equal )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ 5, 5 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( edgeCases, alreadySorted )
{
   const int size = 1024;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( edgeCases, reverseSorted )
{
   const int size = 1024;
   std::vector< int > arr( size );
   std::iota( arr.rbegin(), arr.rend(), 0 );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( edgeCases, allIdentical )
{
   const int size = 1024;
   std::vector< int > arr( size, 42 );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Power-of-2 and non-power-of-2 size tests
// =============================================================================

class PowerOfTwoSizes : public ::testing::TestWithParam< int >
{};

TEST_P( PowerOfTwoSizes, sortRandomData )
{
   const int size = GetParam();
   std::vector< int > arr( size );
   std::mt19937 rng( size );
   std::generate(
      arr.begin(),
      arr.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}

INSTANTIATE_TEST_SUITE_P(
   bitonicSort,
   PowerOfTwoSizes,
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

class NonPowerOfTwoSizes : public ::testing::TestWithParam< int >
{};

TEST_P( NonPowerOfTwoSizes, sortRandomData )
{
   const int size = GetParam();
   std::vector< int > arr( size );
   std::mt19937 rng( size + 12345 );
   std::generate(
      arr.begin(),
      arr.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size " << size;
}

INSTANTIATE_TEST_SUITE_P(
   bitonicSort,
   NonPowerOfTwoSizes,
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
// Permutation tests (exhaustive for small sizes)
// =============================================================================

TEST( permutations, allPermutationSize_2_to_7 )
{
   for( int i = 2; i <= 7; i++ ) {
      int size = i;
      std::vector< int > orig( size );
      std::iota( orig.begin(), orig.end(), 0 );

      do {
         TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( orig );
         auto view = cudaArr.getView();

         BitonicSort::sort( view );

         EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed " << i << '\n';
      } while( std::next_permutation( orig.begin(), orig.end() ) );
   }
}

TEST( permutations, sampledPermutationSize8 )
{
   int size = 8;
   const int stride = 151;
   int i = 0;

   std::vector< int > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      if( ( i++ ) % stride != 0 )
         continue;

      TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( orig );
      auto view = cudaArr.getView();

      BitonicSort::sort( view );

      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

TEST( permutations, sampledPermutationSize9 )
{
   int size = 9;
   const int stride = 227;
   int i = 0;

   std::vector< int > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      if( ( i++ ) % stride != 0 )
         continue;

      TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( orig );
      auto view = cudaArr.getView();

      BitonicSort::sort( view );

      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

// =============================================================================
// Value type tests
// =============================================================================

TEST( valueTypes, int32 )
{
   std::vector< int > arr( 1000 );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( valueTypes, int64 )
{
   std::vector< std::int64_t > arr( 1000 );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< std::int64_t, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( valueTypes, uint8 )
{
   std::vector< std::uint8_t > arr( 256 );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< std::uint8_t, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( valueTypes, float32 )
{
   TNL::Containers::Array< float, TNL::Devices::Cuda > cudaArr{ 5.0f, 9.4f, 4.6f, 8.9f, 6.2f, 1.15184f, 2.23f };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
}

TEST( valueTypes, float64 )
{
   TNL::Containers::Array< double, TNL::Devices::Cuda > cudaArr{ 5.0, 9.4, 4.6, 8.9, 6.2, 1.15184, 2.23 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
}

// =============================================================================
// Custom struct tests
// =============================================================================

struct SmallStruct
{
   uint8_t m_data[ 6 ];
   __cuda_callable__
   SmallStruct()
   {
      m_data[ 0 ] = 0;
   }
   __cuda_callable__
   SmallStruct( int first )
   {
      m_data[ 0 ] = first;
   }
   __cuda_callable__
   bool
   operator<( const SmallStruct& other ) const
   {
      return m_data[ 0 ] < other.m_data[ 0 ];
   }
   __cuda_callable__
   SmallStruct&
   operator=( const SmallStruct& other )
   {
      m_data[ 0 ] = other.m_data[ 0 ];
      return *this;
   }
};

TEST( customStructs, smallStruct )
{
   TNL::Containers::Array< SmallStruct, TNL::Devices::Cuda > cudaArr{
      SmallStruct( 5 ), SmallStruct( 6 ), SmallStruct( 9 ), SmallStruct( 1 )
   };
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

struct MediumStruct
{
   uint8_t m_data[ 64 ];
   __cuda_callable__
   MediumStruct()
   {
      m_data[ 0 ] = 0;
   }
   __cuda_callable__
   MediumStruct( int first )
   {
      m_data[ 0 ] = first;
   }
   __cuda_callable__
   bool
   operator<( const MediumStruct& other ) const
   {
      return m_data[ 0 ] < other.m_data[ 0 ];
   }
   __cuda_callable__
   MediumStruct&
   operator=( const MediumStruct& other )
   {
      m_data[ 0 ] = other.m_data[ 0 ];
      return *this;
   }
};

TEST( customStructs, mediumStruct )
{
   std::mt19937 rng( 61513 );
   int size = rng() % ( 1 << 15 );
   std::vector< MediumStruct > vec( size );
   for( auto& x : vec )
      x = MediumStruct( rng() );

   TNL::Containers::Array< MediumStruct, TNL::Devices::Cuda > cudaArr( vec );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

struct LargeStruct
{
   uint8_t m_data[ 128 ];
   __cuda_callable__
   LargeStruct()
   {
      m_data[ 0 ] = 0;
   }
   __cuda_callable__
   LargeStruct( int first )
   {
      m_data[ 0 ] = first;
   }
   __cuda_callable__
   bool
   operator<( const LargeStruct& other ) const
   {
      return m_data[ 0 ] < other.m_data[ 0 ];
   }
   __cuda_callable__
   LargeStruct&
   operator=( const LargeStruct& other )
   {
      m_data[ 0 ] = other.m_data[ 0 ];
      return *this;
   }
};

TEST( customStructs, largeStruct )
{
   std::mt19937 rng( 98451 );
   int size = rng() % ( 1 << 14 );
   std::vector< LargeStruct > vec( size );
   for( auto& x : vec )
      x = LargeStruct( rng() );

   TNL::Containers::Array< LargeStruct, TNL::Devices::Cuda > cudaArr( vec );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Custom comparator tests
// =============================================================================

void
sortDescending( TNL::Containers::ArrayView< int, TNL::Devices::Cuda > view )
{
   auto cmpDescending = [] __cuda_callable__( int a, int b )
   {
      return a > b;
   };
   BitonicSort::sort( view, cmpDescending );
}

TEST( customComparator, descending )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ 6, 9, 4, 2, 3 };
   auto view = cudaArr.getView();
   sortDescending( view );

   EXPECT_TRUE( Algorithms::isDescending( view ) ) << "result " << view << '\n';
}

TEST( customComparator, descending_large )
{
   const int size = 10000;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   sortDescending( view );
   EXPECT_TRUE( Algorithms::isDescending( view ) );
}

struct AbsCompare
{
   __cuda_callable__
   bool
   operator()( int a, int b ) const
   {
      return std::abs( a ) < std::abs( b );
   }
};

TEST( customComparator, absCompare )
{
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr{ -5, 3, -2, 8, -1, 4 };
   auto view = cudaArr.getView();
   BitonicSort::sort( view, AbsCompare{} );

   std::vector< int > expected = { -1, -2, 3, 4, -5, 8 };
   for( int i = 0; i < 6; i++ )
      EXPECT_EQ( view.getElement( i ), expected[ i ] );
}

// =============================================================================
// Fetch-and-swap interface tests
// =============================================================================

template< typename ValueType >
void
fetchAndSwapSorter( TNL::Containers::ArrayView< ValueType, TNL::Devices::Cuda > view )
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
   bitonicSort( Index{ 0 }, view.getSize(), Cmp, Swap );
}

TEST( fetchAndSwap, smallArray )
{
   int size = 9;
   const int stride = 227;
   int i = 0;

   std::vector< int > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      if( ( i++ ) % stride != 0 )
         continue;

      TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( orig );
      auto view = cudaArr.getView();
      fetchAndSwapSorter( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

TEST( fetchAndSwap, typeDouble )
{
   int size = 5;
   std::vector< double > orig( size );
   std::iota( orig.begin(), orig.end(), 0 );

   do {
      TNL::Containers::Array< double, TNL::Devices::Cuda > cudaArr( orig );
      auto view = cudaArr.getView();
      fetchAndSwapSorter( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) ) << "result " << view << '\n';
   } while( std::next_permutation( orig.begin(), orig.end() ) );
}

void
fetchAndSwap_sortMiddle( TNL::Containers::ArrayView< int, TNL::Devices::Cuda > view, int from, int to )
{
   auto Cmp = [ = ] __cuda_callable__( int i, int j )
   {
      return view[ i ] < view[ j ];
   };
   auto Swap = [ = ] __cuda_callable__( int i, int j ) mutable
   {
      TNL::swap( view[ i ], view[ j ] );
   };
   bitonicSort( from, to, Cmp, Swap );
}

TEST( fetchAndSwap, subrange )
{
   std::vector< int > orig{ 5, 9, 4, 54, 21, 6, 7, 9, 0, 9, 42, 4 };
   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( orig );
   auto view = cudaArr.getView();
   int from = 3;
   int to = 8;

   fetchAndSwap_sortMiddle( view, from, to );
   EXPECT_TRUE( Algorithms::isAscending( view.getView( 3, 8 ) ) ) << "result " << view << '\n';

   for( std::size_t i = 0; i < orig.size(); i++ ) {
      if( i < static_cast< std::size_t >( from ) || i >= static_cast< std::size_t >( to ) )
         EXPECT_EQ( view.getElement( i ), orig[ i ] );
   }
}

TEST( fetchAndSwap, largeArray )
{
   const int size = 1 << 20;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 12345 ) );

   TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   fetchAndSwapSorter( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Index type tests
// =============================================================================

TEST( indexTypes, int32_index )
{
   using Index = int;
   const Index size = 10000;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::Cuda, Index > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( indexTypes, int64_index )
{
   using Index = std::int64_t;
   const Index size = 10000;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::Cuda, Index > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( indexTypes, size_t_index )
{
   using Index = std::size_t;
   const Index size = 10000;
   std::vector< int > arr( size );
   std::iota( arr.begin(), arr.end(), 0 );
   std::shuffle( arr.begin(), arr.end(), std::mt19937( 42 ) );

   TNL::Containers::Array< int, TNL::Devices::Cuda, Index > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

// =============================================================================
// Large array tests
// =============================================================================

TEST( largeArrays, size1M )
{
   const std::size_t size = 1 << 20;
   std::vector< std::uint32_t > arr( size );
   std::mt19937 rng( size );
   std::generate(
      arr.begin(),
      arr.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< std::uint32_t, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( largeArrays, size16M )
{
   const std::size_t size = 1 << 24;
   std::vector< std::uint32_t > arr( size );
   std::mt19937 rng( size );
   std::generate(
      arr.begin(),
      arr.end(),
      [ & ]()
      {
         return rng();
      } );

   TNL::Containers::Array< std::uint32_t, TNL::Devices::Cuda > cudaArr( arr );
   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) );
}

TEST( largeArrays, size2To32 )
{
   const std::size_t size = 1UL << 32;
   TNL::Containers::Array< std::uint8_t, TNL::Devices::Cuda, std::size_t > cudaArr( size );
   cudaArr.setValue( 0 );
   cudaArr.setElement( 0, 1 );
   cudaArr.setElement( 1, 2 );
   cudaArr.setElement( size - 1, 3 );

   auto view = cudaArr.getView();
   BitonicSort::sort( view );
   EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed for size 2^32 = " << size;
}

// =============================================================================
// Stress tests with random data
// =============================================================================

TEST( stressTests, randomSmallArrays )
{
   std::mt19937 rng( 2006 );
   for( int i = 0; i < 100; i++ ) {
      int size = rng() % ( 1 << 10 ) + 1;
      std::vector< int > arr( size );
      std::generate(
         arr.begin(),
         arr.end(),
         [ & ]()
         {
            return rng();
         } );

      TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
      auto view = cudaArr.getView();
      BitonicSort::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   }
}

TEST( stressTests, randomMediumArrays )
{
   std::mt19937 rng( 304 );
   for( int i = 0; i < 20; i++ ) {
      int size = ( rng() % ( 1 << 18 ) ) + 1;
      std::vector< int > arr( size );
      std::generate(
         arr.begin(),
         arr.end(),
         [ & ]()
         {
            return rng();
         } );

      TNL::Containers::Array< int, TNL::Devices::Cuda > cudaArr( arr );
      auto view = cudaArr.getView();
      BitonicSort::sort( view );
      EXPECT_TRUE( Algorithms::isAscending( view ) );
   }
}

#endif

#include "../../main.h"
