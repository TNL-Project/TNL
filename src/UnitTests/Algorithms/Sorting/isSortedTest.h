#include <TNL/Algorithms/sort.h>
#include <TNL/Containers/Array.h>

#if defined( __HIP__ )
   #include <TNL/Allocators/HipHost.h>
#else
   #include <TNL/Allocators/CudaHost.h>
#endif

#include "gtest/gtest.h"
#include <type_traits>

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Array >
class IsSortedTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which IsSortedTest is instantiated
using ArrayTypes = ::testing::Types<  //
   Array< int, Devices::Sequential, int >,
   Array< double, Devices::Sequential, int >,
   Array< std::uint8_t, Devices::Sequential, std::size_t >,
   Array< int, Devices::Host, int >,
   Array< double, Devices::Host, int >,
   Array< std::uint8_t, Devices::Host, std::size_t >
#if defined( __CUDACC__ )
   ,
   Array< int, Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   Array< std::uint8_t, Devices::Cuda, std::size_t >
#endif
#if defined( __HIP__ )
   ,
   Array< int, Devices::Hip, int >,
   Array< double, Devices::Hip, int >,
   Array< std::uint8_t, Devices::Hip, std::size_t >
#endif
   >;

TYPED_TEST_SUITE( IsSortedTest, ArrayTypes );

// Helper functions that can use __cuda_callable__ lambdas
template< typename ArrayType >
void
test_isSorted_emptyArray()
{
   ArrayType array;
   EXPECT_TRUE( isSorted( array, std::less<>{} ) );
   EXPECT_TRUE( isAscending( array ) );
   EXPECT_TRUE( isDescending( array ) );
}

template< typename ArrayType >
void
test_isSorted_singleElement()
{
   ArrayType array{ 42 };
   EXPECT_TRUE( isSorted( array, std::less<>{} ) );
   EXPECT_TRUE( isAscending( array ) );
   EXPECT_TRUE( isDescending( array ) );
}

template< typename ArrayType >
void
test_isSorted_alreadySorted()
{
   ArrayType array{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   EXPECT_TRUE( isSorted( array, std::less<>{} ) );
   EXPECT_TRUE( isAscending( array ) );
   EXPECT_FALSE( isDescending( array ) );
}

template< typename ArrayType >
void
test_isSorted_reverseSorted()
{
   ArrayType array{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
   EXPECT_TRUE( isSorted( array, std::greater<>{} ) );
   EXPECT_FALSE( isAscending( array ) );
   EXPECT_TRUE( isDescending( array ) );
}

template< typename ArrayType >
void
test_isSorted_withDuplicates()
{
   ArrayType array{ 1, 2, 2, 3, 3, 3, 4, 5, 5, 5 };
   EXPECT_TRUE( isSorted( array, std::less<>{} ) );
   EXPECT_TRUE( isAscending( array ) );
   EXPECT_FALSE( isDescending( array ) );
}

template< typename Array >
void
checkIsSorted( Array& array, int exp, typename Array::IndexType size )
{
   array.setValue( 0 );
   EXPECT_TRUE( isSorted( array, std::less<>{} ) ) << "failed for size 2^" << exp << " + 1 = " << size;
   EXPECT_TRUE( isAscending( array ) ) << "failed for size 2^" << exp << " + 1 = " << size;
   EXPECT_TRUE( isDescending( array ) ) << "failed for size 2^" << exp << " + 1 = " << size;
}

template< typename ArrayType >
void
test_isSorted_powersOfTwoSizes()
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;

   // For GPU devices, size will be max 2^32 + 1 for 64-bit IndexType and 2^16 + 1 for 32-bit IndexType.
   // For host and sequential, it is just 2^16 + 1 to keep the test reasonably fast.
   // (the +1 is because isSorted performs size-1 comparisons)
   constexpr int max_exp = std::is_same_v< DeviceType, Devices::GPU > ? sizeof( IndexType ) * 4 : sizeof( int ) * 4;
   // sizes above this threshold are always allocated in page-locked host memory
   constexpr std::size_t pinnedMemoryThreshold = 1UL << 30;
   for( int exp = max_exp; exp >= 0; exp -= 4 ) {
      const IndexType size = ( static_cast< IndexType >( 1 ) << exp ) + 1;
      const bool bigAllocation = static_cast< std::size_t >( size ) * sizeof( ValueType ) >= pinnedMemoryThreshold;
      if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
         // The big allocation needs several GiB, which may not be free on shared CI runners.
         // The test makes a single pass and needs no scratch buffers,
         // so the same kernels can process page-locked host memory mapped into the device address space.
         if( bigAllocation ) {
#if defined( __HIP__ )
            using PinnedAllocator = Allocators::HipHost< ValueType >;
#else
            using PinnedAllocator = Allocators::CudaHost< ValueType >;
#endif
            Array< ValueType, DeviceType, IndexType, PinnedAllocator > array( size );
            checkIsSorted( array, exp, size );
            continue;
         }
      }
      ArrayType array( size );
      checkIsSorted( array, exp, size );
   }
}

// TYPED_TEST blocks that call wrapper functions
TYPED_TEST( IsSortedTest, emptyArray )
{
   test_isSorted_emptyArray< typename TestFixture::ArrayType >();
}

TYPED_TEST( IsSortedTest, singleElement )
{
   test_isSorted_singleElement< typename TestFixture::ArrayType >();
}

TYPED_TEST( IsSortedTest, alreadySorted )
{
   test_isSorted_alreadySorted< typename TestFixture::ArrayType >();
}

TYPED_TEST( IsSortedTest, reverseSorted )
{
   test_isSorted_reverseSorted< typename TestFixture::ArrayType >();
}

TYPED_TEST( IsSortedTest, withDuplicates )
{
   test_isSorted_withDuplicates< typename TestFixture::ArrayType >();
}

TYPED_TEST( IsSortedTest, powersOfTwoSizes )
{
   test_isSorted_powersOfTwoSizes< typename TestFixture::ArrayType >();
}

#include "../../main.h"
