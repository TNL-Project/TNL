#include <TNL/DiscreteMath.h>

#ifdef HAVE_GTEST
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
      for( unsigned char b = -16; b < 16; b++ ) {
         EXPECT_FALSE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_FALSE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }
   for( unsigned char a = 16; a <= 32; a++ )
      for( unsigned char b = 16; b < 255; b++ ) {
         EXPECT_TRUE( TNL::integerMultiplyOverflow( a, b ) ) << "a = " << +a << ", b = " << +b;
         EXPECT_TRUE( TNL::integerMultiplyOverflow( b, a ) ) << "a = " << +a << ", b = " << +b;
      }
}

#endif

#include "main.h"
