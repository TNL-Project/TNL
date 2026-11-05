#pragma once

#include <stdexcept>

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/uncompress.h>
#include <TNL/Containers/Vector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Vector >
class UncompressTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
};

// types for which UncompressTest is instantiated
using VectorTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   Vector< int, Devices::Sequential, int >,
   Vector< int, Devices::Sequential, long >,
   Vector< long, Devices::Sequential, int >,
   Vector< long, Devices::Sequential, long >,

   Vector< int, Devices::Host, int >,
   Vector< int, Devices::Host, long >,
   Vector< long, Devices::Host, int >,
   Vector< long, Devices::Host, long >
#elif defined( __CUDACC__ )
   Vector< int, Devices::Cuda, int >,
   Vector< int, Devices::Cuda, long >,
   Vector< long, Devices::Cuda, int >,
   Vector< long, Devices::Cuda, long >
#elif defined( __HIP__ )
   Vector< int, Devices::Hip, int >,
   Vector< int, Devices::Hip, long >,
   Vector< long, Devices::Hip, int >,
   Vector< long, Devices::Hip, long >
#endif
   >;

TYPED_TEST_SUITE( UncompressTest, VectorTypes );

// ---------------------------------------------------------------------------
// clang-format off
// Basic: indices {0,5,7,8}, maskSize=11
//                  0 1 2 3 4 5 6 7 8 9 10
//   expected mask:{1,0,0,0,0,1,0,1,1,0, 0 }
// clang-format on
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_basic()
{
   using IndexType = typename VectorType::IndexType;

   // clang-format off
   VectorType indices{ 0, 5, 7, 8 };
   // clang-format on

   auto mask = uncompress< VectorType >( indices, IndexType( 11 ) );

   ASSERT_EQ( mask.getSize(), IndexType( 11 ) );
   EXPECT_EQ( mask.getElement( 0 ), 1 );
   EXPECT_EQ( mask.getElement( 1 ), 0 );
   EXPECT_EQ( mask.getElement( 2 ), 0 );
   EXPECT_EQ( mask.getElement( 3 ), 0 );
   EXPECT_EQ( mask.getElement( 4 ), 0 );
   EXPECT_EQ( mask.getElement( 5 ), 1 );
   EXPECT_EQ( mask.getElement( 6 ), 0 );
   EXPECT_EQ( mask.getElement( 7 ), 1 );
   EXPECT_EQ( mask.getElement( 8 ), 1 );
   EXPECT_EQ( mask.getElement( 9 ), 0 );
   EXPECT_EQ( mask.getElement( 10 ), 0 );
}

TYPED_TEST( UncompressTest, basic )
{
   UncompressTest_basic< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Fill overload: same result written into an existing vector.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_fill_overload()
{
   using IndexType = typename VectorType::IndexType;

   VectorType indices{ 0, 5, 7, 8 };
   VectorType mask;
   uncompress( indices, mask, IndexType( 11 ) );

   ASSERT_EQ( mask.getSize(), IndexType( 11 ) );
   EXPECT_EQ( mask.getElement( 0 ), 1 );
   EXPECT_EQ( mask.getElement( 5 ), 1 );
   EXPECT_EQ( mask.getElement( 7 ), 1 );
   EXPECT_EQ( mask.getElement( 8 ), 1 );
   EXPECT_EQ( mask.getElement( 1 ), 0 );
   EXPECT_EQ( mask.getElement( 9 ), 0 );
}

TYPED_TEST( UncompressTest, fill_overload )
{
   UncompressTest_fill_overload< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Indices on the host, mask on the device of the tested vector type.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_host_indices()
{
   using IndexType = typename VectorType::IndexType;
   using HostVectorType = Vector< typename VectorType::ValueType, Devices::Host, IndexType >;

   HostVectorType indices{ 0, 5, 7, 8 };
   VectorType mask;
   uncompress( indices, mask, IndexType( 11 ) );

   ASSERT_EQ( mask.getSize(), IndexType( 11 ) );
   EXPECT_EQ( mask.getElement( 0 ), 1 );
   EXPECT_EQ( mask.getElement( 5 ), 1 );
   EXPECT_EQ( mask.getElement( 7 ), 1 );
   EXPECT_EQ( mask.getElement( 8 ), 1 );
   EXPECT_EQ( mask.getElement( 1 ), 0 );
   EXPECT_EQ( mask.getElement( 9 ), 0 );
}

TYPED_TEST( UncompressTest, host_indices )
{
   UncompressTest_host_indices< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Auto-detect mask size: maskSize == 0, max(indices) + 1 = 9.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_auto_mask_size()
{
   using IndexType = typename VectorType::IndexType;

   VectorType indices{ 0, 5, 7, 8 };
   auto mask = uncompress< VectorType >( indices );

   // maskSize should be max(indices)+1 = 8+1 = 9
   ASSERT_EQ( mask.getSize(), IndexType( 9 ) );
   EXPECT_EQ( mask.getElement( 0 ), 1 );
   EXPECT_EQ( mask.getElement( 5 ), 1 );
   EXPECT_EQ( mask.getElement( 7 ), 1 );
   EXPECT_EQ( mask.getElement( 8 ), 1 );
   EXPECT_EQ( mask.getElement( 1 ), 0 );
   EXPECT_EQ( mask.getElement( 6 ), 0 );
}

TYPED_TEST( UncompressTest, auto_mask_size )
{
   UncompressTest_auto_mask_size< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Empty index vector with explicit mask size: mask should be all zeros.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_empty_input_explicit_size()
{
   using IndexType = typename VectorType::IndexType;

   VectorType indices;  // empty
   auto mask = uncompress< VectorType >( indices, IndexType( 5 ) );

   ASSERT_EQ( mask.getSize(), IndexType( 5 ) );
   for( IndexType i = 0; i < 5; ++i )
      EXPECT_EQ( mask.getElement( i ), 0 );
}

TYPED_TEST( UncompressTest, empty_input_explicit_size )
{
   UncompressTest_empty_input_explicit_size< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Empty index vector with auto mask size: result should be empty mask.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_empty_input_auto_size()
{
   using IndexType = typename VectorType::IndexType;

   VectorType indices;  // empty
   auto mask = uncompress< VectorType >( indices );

   EXPECT_EQ( mask.getSize(), IndexType( 0 ) );
}

TYPED_TEST( UncompressTest, empty_input_auto_size )
{
   UncompressTest_empty_input_auto_size< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Round-trip: compress then uncompress reproduces the original mask.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_roundtrip()
{
   using IndexType = typename VectorType::IndexType;

   // clang-format off
   //                   0  1  2  3  4  5  6  7  8  9 10
   VectorType original{ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
   // clang-format on
   auto indices = TNL::Algorithms::compress< VectorType >( original );
   auto restored = uncompress< VectorType >( indices, IndexType( 11 ) );

   ASSERT_EQ( restored.getSize(), IndexType( 11 ) );
   for( IndexType i = 0; i < 11; ++i )
      EXPECT_EQ( restored.getElement( i ), original.getElement( i ) );
}

TYPED_TEST( UncompressTest, roundtrip )
{
   UncompressTest_roundtrip< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Range [begin, end) of the index vector.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_range()
{
   using IndexType = typename VectorType::IndexType;
   using HostVectorType = Vector< typename VectorType::ValueType, Devices::Host, IndexType >;

   // only the first two indices are valid, the rest must be ignored
   // (99 would be out of range for the mask size 11)
   VectorType indices{ 0, 5, 99, 99 };
   VectorType mask( 11, 1 );
   uncompress( indices, mask, IndexType( 11 ), IndexType( 0 ), IndexType( 2 ) );
   ASSERT_EQ( mask.getSize(), IndexType( 11 ) );
   VectorType expected{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
   EXPECT_EQ( mask, expected );

   // subrange in the middle of the index vector
   VectorType indices2{ 99, 2, 7, 99 };
   uncompress( indices2, mask, IndexType( 11 ), IndexType( 1 ), IndexType( 3 ) );
   VectorType expected2{ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 };
   EXPECT_EQ( mask, expected2 );

   // end == 0 means the end of the index vector
   VectorType indices3{ 99, 3, 4 };
   uncompress( indices3, mask, IndexType( 11 ), IndexType( 1 ) );
   VectorType expected3{ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0 };
   EXPECT_EQ( mask, expected3 );

   // auto-detected mask size uses only the indices in the range
   auto mask2 = uncompress< VectorType >( indices2, IndexType( 0 ), IndexType( 1 ), IndexType( 3 ) );
   EXPECT_EQ( mask2.getSize(), IndexType( 8 ) );
   EXPECT_EQ( mask2.getElement( 2 ), 1 );
   EXPECT_EQ( mask2.getElement( 7 ), 1 );

   // indices on the host, only the range is used
   HostVectorType hostIndices{ 99, 2, 7, 99 };
   uncompress( hostIndices, mask, IndexType( 11 ), IndexType( 1 ), IndexType( 3 ) );
   EXPECT_EQ( mask, expected2 );
}

TYPED_TEST( UncompressTest, range )
{
   UncompressTest_range< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// uncompressUnchecked gives the same results as uncompress for valid indices.
// ---------------------------------------------------------------------------

template< typename VectorType >
void
UncompressTest_unchecked()
{
   using IndexType = typename VectorType::IndexType;
   using HostVectorType = Vector< typename VectorType::ValueType, Devices::Host, IndexType >;

   // returning overload
   VectorType indices{ 0, 5, 7, 8 };
   auto mask = uncompressUnchecked< VectorType >( indices, IndexType( 11 ) );
   VectorType expected{ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
   EXPECT_EQ( mask, expected );

   // auto-detected mask size
   auto mask2 = uncompressUnchecked< VectorType >( indices );
   EXPECT_EQ( mask2.getSize(), IndexType( 9 ) );
   EXPECT_EQ( mask2, uncompress< VectorType >( indices ) );

   // filling overload: the previous content of the mask is overwritten
   VectorType mask3( 11, 1 );
   uncompressUnchecked( indices, mask3, IndexType( 11 ) );
   EXPECT_EQ( mask3, expected );

   // duplicate indices
   VectorType duplicates{ 5, 0, 5, 8, 7, 0 };
   uncompressUnchecked( duplicates, mask3, IndexType( 11 ) );
   EXPECT_EQ( mask3, expected );

   // range: indices out of the range are ignored
   VectorType indices2{ 99, 2, 7, 99 };
   uncompressUnchecked( indices2, mask3, IndexType( 11 ), IndexType( 1 ), IndexType( 3 ) );
   VectorType expected2{ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 };
   EXPECT_EQ( mask3, expected2 );

   // indices on the host
   HostVectorType hostIndices{ 99, 2, 7, 99 };
   VectorType mask4;
   uncompressUnchecked( hostIndices, mask4, IndexType( 11 ), IndexType( 1 ), IndexType( 3 ) );
   EXPECT_EQ( mask4, expected2 );

   // empty input
   VectorType empty;
   EXPECT_EQ( uncompressUnchecked< VectorType >( empty ).getSize(), IndexType( 0 ) );
   uncompressUnchecked( empty, mask4, IndexType( 11 ) );
   EXPECT_EQ( mask4, VectorType( 11, 0 ) );
}

TYPED_TEST( UncompressTest, unchecked )
{
   UncompressTest_unchecked< typename TestFixture::VectorType >();
}

// ---------------------------------------------------------------------------
// Invalid index throws std::invalid_argument.
// (Only testable on host because CUDA kernels cannot throw exceptions.)
// ---------------------------------------------------------------------------

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )

template< typename VectorType >
void
UncompressTest_invalid_index()
{
   using IndexType = typename VectorType::IndexType;

   VectorType indices{ 0, 3, 99 };  // index 99 is out of range for maskSize=10
   EXPECT_THROW( uncompress< VectorType >( indices, IndexType( 10 ) ), std::invalid_argument );
}

TEST( UncompressHostTest, invalid_index )
{
   using VectorType = Vector< int, Devices::Host, int >;
   UncompressTest_invalid_index< VectorType >();
}

#endif

#include "../main.h"
