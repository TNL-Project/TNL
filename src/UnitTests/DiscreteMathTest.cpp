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

TEST( DiscreteMathTest, primeFactorizationTest )
{
   EXPECT_TRUE( primeFactorization( 0 ).empty() );
   EXPECT_EQ( primeFactorization( 1 ), ( std::vector< int >{ 1 } ) );
   EXPECT_EQ( primeFactorization( 2 ), ( std::vector< int >{ 2 } ) );
   EXPECT_EQ( primeFactorization( 3 ), ( std::vector< int >{ 3 } ) );
   EXPECT_EQ( primeFactorization( 4 ), ( std::vector< int >{ 2, 2 } ) );
   EXPECT_EQ( primeFactorization( 5 ), ( std::vector< int >{ 5 } ) );
   EXPECT_EQ( primeFactorization( 6 ), ( std::vector< int >{ 2, 3 } ) );
   EXPECT_EQ( primeFactorization( 7 ), ( std::vector< int >{ 7 } ) );
   EXPECT_EQ( primeFactorization( 8 ), ( std::vector< int >{ 2, 2, 2 } ) );
   EXPECT_EQ( primeFactorization( 9 ), ( std::vector< int >{ 3, 3 } ) );
   EXPECT_EQ( primeFactorization( 10 ), ( std::vector< int >{ 2, 5 } ) );
   EXPECT_EQ( primeFactorization( 11 ), ( std::vector< int >{ 11 } ) );
   EXPECT_EQ( primeFactorization( 12 ), ( std::vector< int >{ 2, 2, 3 } ) );
   EXPECT_EQ( primeFactorization( 13 ), ( std::vector< int >{ 13 } ) );
   EXPECT_EQ( primeFactorization( 14 ), ( std::vector< int >{ 2, 7 } ) );
   EXPECT_EQ( primeFactorization( 15 ), ( std::vector< int >{ 3, 5 } ) );
   EXPECT_EQ( primeFactorization( 16 ), ( std::vector< int >{ 2, 2, 2, 2 } ) );
   EXPECT_EQ( primeFactorization( 17 ), ( std::vector< int >{ 17 } ) );
   EXPECT_EQ( primeFactorization( 18 ), ( std::vector< int >{ 2, 3, 3 } ) );
   EXPECT_EQ( primeFactorization( 19 ), ( std::vector< int >{ 19 } ) );
   EXPECT_EQ( primeFactorization( 20 ), ( std::vector< int >{ 2, 2, 5 } ) );

   EXPECT_EQ( primeFactorization( 864 ), ( std::vector< int >{ 2, 2, 2, 2, 2, 3, 3, 3 } ) );
   EXPECT_EQ( primeFactorization( 20460 ), ( std::vector< int >{ 2, 2, 3, 5, 11, 31 } ) );

   for( int n = 21; n <= 1024; n++ ) {
      const std::vector< int > prime_factors = primeFactorization( n );
      int product = 1;
      for( auto prime : prime_factors )
         product *= prime;
      EXPECT_EQ( n, product );
   }
}

TEST( DiscreteMathTest, cartesianPower )
{
   using result_t = std::set< std::vector< int > >;
   std::vector< int > array_1 = { 0 };
   std::vector< int > array_2 = { 0, 1 };
   std::vector< int > array_3 = { 0, 1, 2 };

   // N = 1
   EXPECT_EQ( cartesianPower( array_1, 1 ), ( result_t{ { 0 } } ) );
   EXPECT_EQ( cartesianPower( array_2, 1 ), ( result_t{ { 0 }, { 1 } } ) );
   EXPECT_EQ( cartesianPower( array_3, 1 ), ( result_t{ { 0 }, { 1 }, { 2 } } ) );

   // N = 2
   EXPECT_EQ( cartesianPower( array_1, 2 ), ( result_t{ { 0, 0 } } ) );
   EXPECT_EQ( cartesianPower( array_2, 2 ), ( result_t{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } } ) );
   EXPECT_EQ( cartesianPower( array_3, 2 ),
              ( result_t{ { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 0 }, { 1, 1 }, { 1, 2 }, { 2, 0 }, { 2, 1 }, { 2, 2 } } ) );

   // N = 3
   EXPECT_EQ( cartesianPower( array_1, 3 ), ( result_t{ { 0, 0, 0 } } ) );
   EXPECT_EQ(
      cartesianPower( array_2, 3 ),
      ( result_t{ { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } } ) );
   EXPECT_EQ( cartesianPower( array_3, 3 ),
              ( result_t{ { 0, 0, 0 }, { 0, 0, 1 }, { 0, 0, 2 }, { 0, 1, 0 }, { 0, 1, 1 }, { 0, 1, 2 }, { 0, 2, 0 },
                          { 0, 2, 1 }, { 0, 2, 2 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 0, 2 }, { 1, 1, 0 }, { 1, 1, 1 },
                          { 1, 1, 2 }, { 1, 2, 0 }, { 1, 2, 1 }, { 1, 2, 2 }, { 2, 0, 0 }, { 2, 0, 1 }, { 2, 0, 2 },
                          { 2, 1, 0 }, { 2, 1, 1 }, { 2, 1, 2 }, { 2, 2, 0 }, { 2, 2, 1 }, { 2, 2, 2 } } ) );
}

TEST( DiscreteMathTest, integerFactorizationTuples_pairs )
{
   using pairs_t = std::set< std::array< int, 2 > >;

   EXPECT_TRUE( integerFactorizationTuples< 2 >( 0 ).empty() );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 1 ), ( pairs_t{ { 1, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 2 ), ( pairs_t{ { 1, 2 }, { 2, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 3 ), ( pairs_t{ { 1, 3 }, { 3, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 4 ), ( pairs_t{ { 1, 4 }, { 2, 2 }, { 4, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 5 ), ( pairs_t{ { 1, 5 }, { 5, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 2 >( 6 ), ( pairs_t{ { 1, 6 }, { 2, 3 }, { 3, 2 }, { 6, 1 } } ) );
}

TEST( DiscreteMathTest, integerFactorizationTuples_triplets )
{
   using triplets_t = std::set< std::array< int, 3 > >;

   EXPECT_TRUE( integerFactorizationTuples< 3 >( 0 ).empty() );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 1 ), ( triplets_t{ { 1, 1, 1 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 2 ), ( triplets_t{ { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 3 ), ( triplets_t{ { 3, 1, 1 }, { 1, 3, 1 }, { 1, 1, 3 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 4 ),
              ( triplets_t{ { 4, 1, 1 }, { 1, 4, 1 }, { 1, 1, 4 }, { 2, 2, 1 }, { 2, 1, 2 }, { 1, 2, 2 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 6 ),
              ( triplets_t{ { 6, 1, 1 },
                            { 1, 6, 1 },
                            { 1, 1, 6 },
                            { 2, 3, 1 },
                            { 3, 2, 1 },
                            { 2, 1, 3 },
                            { 3, 1, 2 },
                            { 1, 3, 2 },
                            { 1, 2, 3 } } ) );
   EXPECT_EQ( integerFactorizationTuples< 3 >( 12 ),
              ( triplets_t{
                 { 12, 1, 1 },
                 { 1, 12, 1 },
                 { 1, 1, 12 },
                 { 4, 3, 1 },
                 { 3, 4, 1 },
                 { 4, 1, 3 },
                 { 3, 1, 4 },
                 { 1, 4, 3 },
                 { 1, 3, 4 },
                 { 3, 2, 2 },
                 { 2, 3, 2 },
                 { 2, 2, 3 },
                 { 6, 2, 1 },
                 { 2, 6, 1 },
                 { 6, 1, 2 },
                 { 2, 1, 6 },
                 { 1, 6, 2 },
                 { 1, 2, 6 },
              } ) );
}

#include "main.h"
