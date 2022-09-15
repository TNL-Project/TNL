#pragma once

#ifdef HAVE_GTEST
#include <type_traits>
#include <bitset>
#include <cstdint>

#include <TNL/Containers/BitArray.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Base >
class BitArrayTest : public ::testing::Test
{
protected:
   using BaseType = Base;
};

// types for which BitArrayTest is instantiated
using BaseTypes = ::testing::Types<
   typename BitArray< 1 >::BaseType
>;

TYPED_TEST_SUITE( BitArrayTest, BaseTypes );


TYPED_TEST( BitArrayTest, constructorWithNoParams )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 1, BaseType > b1;
   EXPECT_EQ( b1.getSize(), 1 );
   for( int i = 0; i < 1; i++ )
      EXPECT_EQ( b1[ i ], 0 );

   BitArray< 8, BaseType > b8;
   EXPECT_EQ( b8.getSize(), 8 );
   for( int i = 0; i < 8; i++ )
      EXPECT_EQ( b8[ i ], 0 );

   BitArray< 30, BaseType > b30;
   EXPECT_EQ( b30.getSize(), 30 );
   for( int i = 0; i < 30; i++ )
      EXPECT_EQ( b30[ i ], 0 );
}

TYPED_TEST( BitArrayTest, constructorWithData )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 1, BaseType > b1{ 0b1 };
   std::bitset< 1 > std_b1{ 0b1 };
   for( int i = 0; i < 1; i++ )
      EXPECT_EQ( b1[ i ], std_b1[ i ] );

   BitArray< 8, BaseType > b8{ 0b10101100 };
   std::bitset< 8 > std_b8{ 0b10101100 };
   for( int i = 0; i < 8; i++ )
      EXPECT_EQ( b8[ i ], std_b8[ i ] );

   BitArray< 30, BaseType > b30{ 0b111000111000111000111000111000 };
   std::bitset< 30 > std_b30{ 0b111000111000111000111000111000 };
   for( int i = 0; i < 30; i++ )
      EXPECT_EQ( b30[ i ], std_b30[ i ] );
}

TYPED_TEST( BitArrayTest, constructorWithArray )
{
   using BaseType = std::uint8_t;

   BaseType data8[]{ 0b11001100, 0b10001101 };

   BitArray< 16, BaseType > b16{ data8, 2 };
   std::bitset< 16 > std_b16{ 0b1100110010001101 };

   for( int i = 0; i < 16; i++ )
      EXPECT_EQ( b16[ i ], std_b16[ i ] );
}

TYPED_TEST( BitArrayTest, copyConstructor )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 30, BaseType > b30{ 0b111000111000111000111000111000 };
   BitArray< 30, BaseType > copy_b30{ b30 };
   for( int i = 0; i < 30; i++ )
      EXPECT_EQ( b30[ i ], copy_b30[ i ] );
}

TYPED_TEST( BitArrayTest, assignmentOperator )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 30, BaseType > b30{ 0b111000111000111000111000111000 };
   BitArray< 30, BaseType > copy_b30;
   copy_b30 = b30;
   for( int i = 0; i < 30; i++ )
      EXPECT_EQ( b30[ i ], copy_b30[ i ] );
}

TYPED_TEST( BitArrayTest, comparisonOperator )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 30, BaseType > b30{ 0b111000111000111000111000111000 };
   BitArray< 30, BaseType > mod_b30{ 0b011000111000111000111000111000 };
   BitArray< 30, BaseType > copy_b30;
   copy_b30 = b30;

   EXPECT_EQ( b30, copy_b30 );
   EXPECT_NE( b30, mod_b30 );

   uint8_t data8[]{ 0b11001100, 0b10001101 };
   BitArray< 16, std::uint8_t > b1{ data8, 2 };
   BitArray< 16, std::uint16_t > b2{ 0b1100110010001101 };
   EXPECT_EQ( b1, b2 );
}

TYPED_TEST( BitArrayTest, incrementOperator )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 16, std::uint8_t > b1;
   BitArray< 16, std::uint16_t > b2;

   for( int i = 0; i < 1000; i++ )
   {
      b1++;
      b2++;
      EXPECT_EQ( b1, b2 );
   }
}

TYPED_TEST( BitArrayTest, shiftOperator )
{
   using BaseType = typename TestFixture::BaseType;

   BitArray< 30, BaseType > b30{ 0b111000111000111000111000111000 };
   BitArray< 30, BaseType > b30_1{ b30 };
   BitArray< 30, BaseType > b30_2{ b30 };

   BitArray< 30, BaseType > b30_shift_1{ 0b011100011100011100011100011100 };
   BitArray< 30, BaseType > b30_shift_2{ 0b001110001110001110001110001110 };
   BitArray< 30, BaseType > b30_shift_3{ 0b000111000111000111000111000111 };
   BitArray< 30, BaseType > b30_shift_4{ 0b000011100011100011100011100011 };
   BitArray< 30, BaseType > b30_shift_5{ 0b000001110001110001110001110001 };

   b30 >>= 1;
   EXPECT_EQ( b30, b30_shift_1 );
   b30 >>= 1;
   EXPECT_EQ( b30, b30_shift_2 );
   b30 >>= 1;
   EXPECT_EQ( b30, b30_shift_3 );
   b30 >>= 1;
   EXPECT_EQ( b30, b30_shift_4 );
   b30 >>= 1;
   EXPECT_EQ( b30, b30_shift_5 );

   b30_1 >>= 2;
   EXPECT_EQ( b30_1, b30_shift_2 );

   b30_1 >>= 2;
   EXPECT_EQ( b30_1, b30_shift_4 );

   b30_2 >>= 5;
   EXPECT_EQ( b30_2, b30_shift_5 );
}

TYPED_TEST( BitArrayTest, indexOperator )
{
   using BaseType = typename TestFixture::BaseType;

   uint8_t data8[]{ 0b11001100, 0b10001101 };
   BitArray< 16, std::uint8_t > b{ data8, 2 };

   EXPECT_EQ( b[  0 ], 1 );
   EXPECT_EQ( b[  1 ], 0 );
   EXPECT_EQ( b[  2 ], 1 );
   EXPECT_EQ( b[  3 ], 1 );
   EXPECT_EQ( b[  4 ], 0 );
   EXPECT_EQ( b[  5 ], 0 );
   EXPECT_EQ( b[  6 ], 0 );
   EXPECT_EQ( b[  7 ], 1 );
   EXPECT_EQ( b[  8 ], 0 );
   EXPECT_EQ( b[  9 ], 0 );
   EXPECT_EQ( b[ 10 ], 1 );
   EXPECT_EQ( b[ 11 ], 1 );
   EXPECT_EQ( b[ 12 ], 0 );
   EXPECT_EQ( b[ 13 ], 0 );
   EXPECT_EQ( b[ 14 ], 1 );
   EXPECT_EQ( b[ 15 ], 1 );
}

TYPED_TEST( BitArrayTest, reset )
{
   using BaseType = typename TestFixture::BaseType;

   uint8_t data8[]{ 0b11001100, 0b10001101 };
   BitArray< 16, std::uint8_t > b{ data8, 2 };
   b.reset();

   EXPECT_EQ( b[  0 ], 0 );
   EXPECT_EQ( b[  1 ], 0 );
   EXPECT_EQ( b[  2 ], 0 );
   EXPECT_EQ( b[  3 ], 0 );
   EXPECT_EQ( b[  4 ], 0 );
   EXPECT_EQ( b[  5 ], 0 );
   EXPECT_EQ( b[  6 ], 0 );
   EXPECT_EQ( b[  7 ], 0 );
   EXPECT_EQ( b[  8 ], 0 );
   EXPECT_EQ( b[  9 ], 0 );
   EXPECT_EQ( b[ 10 ], 0 );
   EXPECT_EQ( b[ 11 ], 0 );
   EXPECT_EQ( b[ 12 ], 0 );
   EXPECT_EQ( b[ 13 ], 0 );
   EXPECT_EQ( b[ 14 ], 0 );
   EXPECT_EQ( b[ 15 ], 0 );
}


#endif // HAVE_GTEST


#include "../main.h"
