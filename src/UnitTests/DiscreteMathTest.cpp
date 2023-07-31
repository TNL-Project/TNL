#include <TNL/DiscreteMath.h>

#include <gtest/gtest.h>

using namespace TNL;

void
testPower( const size_t value, const size_t power, const size_t expectation )
{
   EXPECT_EQ( TNL::discretePow( value, power ), expectation ) << value << " " << power;
}

TEST( DiscreteMathTest, PowerTest )
{
   testPower( 0, 1, 0 );
   testPower( 1, 1, 1 );

   testPower( 0, 2, 0 );
   testPower( 1, 2, 1 );
   testPower( 2, 2, 4 );

   testPower( 0, 3, 0 );
   testPower( 1, 3, 1 );
   testPower( 2, 3, 8 );
   testPower( 3, 3, 27 );

   testPower( 0, 4, 0 );
   testPower( 1, 4, 1 );
   testPower( 2, 4, 16 );
   testPower( 3, 4, 81 );
   testPower( 4, 4, 256 );
}

TEST( DiscreteMathTest, Log2Test )
{
   static_assert( TNL::discreteLog2( 0 ) == std::numeric_limits< int >::max() );
   static_assert( TNL::discreteLog2( 0UL ) == std::numeric_limits< unsigned long >::max() );

   static_assert( TNL::discreteLog2( 1 ) == 0 );
   static_assert( TNL::discreteLog2( 2 ) == 1 );
   static_assert( TNL::discreteLog2( 3 ) == 1 );
   static_assert( TNL::discreteLog2( 4 ) == 2 );
   static_assert( TNL::discreteLog2( 5 ) == 2 );
   static_assert( TNL::discreteLog2( 6 ) == 2 );
   static_assert( TNL::discreteLog2( 7 ) == 2 );
   static_assert( TNL::discreteLog2( 8 ) == 3 );
   static_assert( TNL::discreteLog2( 9 ) == 3 );
   static_assert( TNL::discreteLog2( 10 ) == 3 );
   static_assert( TNL::discreteLog2( 11 ) == 3 );
   static_assert( TNL::discreteLog2( 12 ) == 3 );
   static_assert( TNL::discreteLog2( 13 ) == 3 );
   static_assert( TNL::discreteLog2( 14 ) == 3 );
   static_assert( TNL::discreteLog2( 15 ) == 3 );
   static_assert( TNL::discreteLog2( 16 ) == 4 );
   static_assert( TNL::discreteLog2( 31 ) == 4 );
   static_assert( TNL::discreteLog2( 32 ) == 5 );

   static_assert( TNL::discreteLog2( 4294967286 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967287 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967288 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967289 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967290 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967291 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967292 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967293 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967294 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967295 ) == 31 );
   static_assert( TNL::discreteLog2( 4294967296 ) == 32 );
}

void
testCombinations( const int k, const int n, const int expectation )
{
   EXPECT_EQ( TNL::combinationsCount( k, n ), expectation ) << k << " " << n;
}

TEST( DiscreteMathTest, CombinationsTest )
{
   testCombinations( 0, 1, 1 );
   testCombinations( 1, 1, 1 );

   testCombinations( 0, 2, 1 );
   testCombinations( 1, 2, 2 );
   testCombinations( 2, 2, 1 );

   testCombinations( 0, 3, 1 );
   testCombinations( 1, 3, 3 );
   testCombinations( 2, 3, 3 );
   testCombinations( 3, 3, 1 );

   testCombinations( 0, 4, 1 );
   testCombinations( 1, 4, 4 );
   testCombinations( 2, 4, 6 );
   testCombinations( 3, 4, 4 );
   testCombinations( 4, 4, 1 );
}

void
testFirstKCombinationsSum( const int k, const int n, const int expectation )
{
   EXPECT_EQ( TNL::firstKCombinationsSum( k, n ), expectation ) << " k = " << k << " n =  " << n;
}

TEST( DiscreteMathTest, firstKCombinationsSumTest )
{
   testFirstKCombinationsSum( 0, 1, 0 );
   testFirstKCombinationsSum( 1, 1, 1 );
   testFirstKCombinationsSum( 2, 1, 2 );

   testFirstKCombinationsSum( 0, 2, 0 );
   testFirstKCombinationsSum( 1, 2, 1 );
   testFirstKCombinationsSum( 2, 2, 3 );
   testFirstKCombinationsSum( 3, 2, 4 );

   testFirstKCombinationsSum( 0, 3, 0 );
   testFirstKCombinationsSum( 1, 3, 1 );
   testFirstKCombinationsSum( 2, 3, 4 );
   testFirstKCombinationsSum( 3, 3, 7 );
   testFirstKCombinationsSum( 4, 3, 8 );

   testFirstKCombinationsSum( 0, 4, 0 );
   testFirstKCombinationsSum( 1, 4, 1 );
   testFirstKCombinationsSum( 2, 4, 5 );
   testFirstKCombinationsSum( 3, 4, 11 );
   testFirstKCombinationsSum( 4, 4, 15 );
   testFirstKCombinationsSum( 5, 4, 16 );
}

TEST( DiscreteMathTest, detectMultiplicationOverflowTest )
{
   for( char a = -7; a <= 8; a++ )
      for( char b = -16; b < 16; b++ ) {
         EXPECT_FALSE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_FALSE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }
   for( char a = 8; a <= 16; a++ )
      for( char b = 16; b < 127; b++ ) {
         EXPECT_TRUE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_TRUE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }

   for( unsigned char a = 0; a <= 16; a++ )
      for( unsigned char b = 0; b < 16; b++ ) {
         EXPECT_FALSE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_FALSE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }
   for( unsigned char a = 16; a <= 32; a++ )
      for( unsigned char b = 16; b < 255; b++ ) {
         EXPECT_TRUE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_TRUE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }
}

#include "main.h"
