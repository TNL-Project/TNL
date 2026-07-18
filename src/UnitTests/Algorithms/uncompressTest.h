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
