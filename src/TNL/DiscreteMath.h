// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include <set>
#include <array>

#include <TNL/Assert.h>

namespace TNL {

/**
 * \brief Divides \e num by \e div and rounds up the result.
 *
 * \param num An integer considered as dividend.
 * \param div An integer considered as divisor.
 */
constexpr int
roundUpDivision( const int num, const int div )
{
   return num / div + static_cast< int >( num % div != 0 );
}

/**
 * \brief Rounds up \e number to the nearest multiple of number \e multiple.
 *
 * \param number Integer we want to round.
 * \param multiple Integer.
 */
constexpr int
roundToMultiple( int number, int multiple )
{
   return multiple * ( number / multiple + static_cast< int >( number % multiple != 0 ) );
}

/**
 * \brief Checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Integer.
 */
constexpr bool
isPow2( int x )
{
   return ( x & ( x - 1 ) ) == 0;
}

/**
 * \brief Checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Long integer.
 */
constexpr bool
isPow2( long int x )
{
   return ( x & ( x - 1 ) ) == 0;
}

/**
 * \brief Computes an integral power of an integer.
 *
 * \param value is the argument of the power function.
 * \param power is the exponent in the power function.
 */
template< typename Index >
constexpr Index
discretePow( Index value, Index power )
{
   Index result = 1;
   for( Index i = 0; i < power; i++ )
      result *= value;
   return result;
}

/**
 * \brief Computes an integral base-2 logarithm.
 *
 * \param value is the argument of the log2 function.
 */
template< typename Index >
constexpr Index
discreteLog2( Index value )
{
   if( value == 0 )
      return std::numeric_limits< Index >::max();
   // algorithm from https://stackoverflow.com/a/994623
   Index result = 0;
   while( value >>= 1 )
      result++;
   return result;
}

/**
 * \brief Computes a product of integers `first * (first+1) * ... * last`.
 *
 * \tparam Index is an integral type used for evaluation of the product.
 * \param first is the first number of the product
 * \param last is the last number of the product.
 */
template< typename Index >
constexpr Index
discreteProduct( Index first, Index last )
{
   Index result = 1;
   for( Index i = first; i <= last; i++ )
      result *= i;
   return result;
}

/**
 * \brief Computes the number of k-combinations in set of n element.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Combination).
 *
 * \tparam Index is an integral type used for the evaluation of the number of combinations.
 * \param k denotes the number of elements in one combination.
 * \param n denotes the number of all elements in the set.
 * \return Number of all k-combinations in set of n elements.
 */
template< typename Index >
constexpr Index
combinationsCount( Index k, Index n )
{
   if( k < 0 )
      return 0;
   return discreteProduct< Index >( k + 1, n ) / discreteProduct< Index >( 1, n - k );
}

/**
 * \brief Compute number of all i-combinations in set of n elements for all i lower or equal to k.
 *
 * \tparam Index is a discrete type used for evaluation of the cumulative number of combinations.
 * \param k denotes maximal number of elements in combination.
 * \param n denotes number of all elements in the set.
 * \return constexpr Index is number of all i-combinations in set of n elements for all i lower or equal to k.
 */
template< typename Index >
constexpr Index
cumulativeCombinationsCount( Index k, Index n )
{
   Index result = 0;
   for( Index i = 0; i <=k; i++ )
      result += combinationsCount( i, n );
   return result;
}

/**
 * \brief Checks if two values of the same integral type can be multiplied without causing integer overflow or underflow.
 *
 * \tparam Index is the integral type of input values.
 * \param a is the first operand in the expression `a * b`
 * \param b is the second operand in the expression `a * b`
 * \return `true` if the operation `a * b` results in an integer overflow or underflow,
 *         and `false` if the result fits into the \e Index type.
 */
template< typename Index, std::enable_if_t< std::is_integral_v< Index >, bool > = true >
bool
integerMultiplyOverflow( Index a, Index b )
{
   if( a == 0 || b == 0 )
      return false;
   const Index result = a * b;
   return a != result / b;
}

/**
 * \brief Calculates the prime factorization of a positive integer.
 *
 * The function uses the simple [trial division](https://en.wikipedia.org/wiki/Trial_division)
 * algorithm, so it is not efficient for large numbers.
 *
 * \tparam Index is the integral type of the input number.
 * \param number is the integer to be factorized.
 * \return A vector of the prime factors.
 */
template< typename Index >
std::vector< Index >
primeFactorization( Index number )
{
   if( number <= 0 )
      return {};
   if( number == 1 )
      return { 1 };

   std::vector< Index > factors;
   Index factor = 2;
   while( number % 2 == 0 ) {
      factors.push_back( factor );
      number /= 2;
   }

   factor = 3;
   while( factor * factor <= number ) {
      if( number % factor == 0 ) {
         factors.push_back( factor );
         number /= factor;
      }
      else {
         factor += 2;
      }
   }

   if( number != 1 )
      factors.push_back( number );

   return factors;
}

/**
 * \brief Calculates the [cartesian power](https://en.wikipedia.org/wiki/Cartesian_product#n-ary_Cartesian_power)
 * of elements in an array: `array^N`.
 *
 * For example, `cartesianPower(std::vector<int>{0,1}, 2)` returns the following set (written in pseudo-code):
 * `{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }`.
 *
 * \tparam T is the type of elements in the \e array.
 * \param array is the input array/vector of elements.
 * \param N is the power of the cartesian product.
 * \return A set of vectors of elements, where each vector contains \e N elements from \e array.
 */
template< typename T >
std::set< std::vector< T > >
cartesianPower( std::vector< T > array, int N )
{
   if( array.empty() || N < 1 )
      return {};
   if( array.size() == 1 ) {
      std::vector< T > tuple( N );
      return { tuple };
   }

   std::set< std::vector< T > > result;

   if( N == 1 ) {
      for( const auto& value : array )
         result.emplace( std::vector< T >{ value } );
      return result;
   }

   std::vector< std::size_t > indices( N, 0 );
   bool changed = true;
   while( changed ) {
      changed = false;
      // add current tuple
      std::vector< T > tuple( N );
      for( int i = 0; i < N; i++ )
         tuple[ i ] = array[ indices[ i ] ];
      result.insert( tuple );
      // loop over the input elements in reverse order
      for( int i = N - 1; ! changed && i >= 0; i-- ) {
         // increment
         indices[ i ]++;
         if( indices[ i ] < array.size() ) {
            // we moved to the next character
            changed = true;
         }
         else {
            // end of string, so roll over
            indices[ i ] = 0;
         }
      }
   }
   return result;
}

/**
 * \brief Finds all possible integer factorizations of a positive integer into
 * a product of `N` factors.
 *
 * \tparam N is the rank of the tuples (2 for pairs, 3 for triplets, etc.)
 * \tparam Index is the integral type of the input number.
 * \param number is the integer to be factorized.
 * \return A set of \e N-tuples, where the product of all components in each
 *         tuple is equal to \e number.
 */
template< std::size_t N, typename Index >
std::set< std::array< Index, N > >
integerFactorizationTuples( Index number )
{
   static_assert( N > 1 );

   if( number < 1 )
      return {};

   // factorize the input
   const std::vector< Index > prime_factors = primeFactorization( number );

   // create object for the result
   std::set< std::array< Index, N > > result;

   // generate component indices
   std::vector< std::uint8_t > identity( N );
   for( std::size_t i = 0; i < N; i++ )
      identity[ i ] = i;
   const auto component_indices = cartesianPower( identity, prime_factors.size() );

   // generate tuples
   for( const auto& components : component_indices ) {
      TNL_ASSERT_EQ( prime_factors.size(), components.size(), "got wrong vector size from the cartesian product" );

      // create default tuple
      std::array< Index, N > tuple;
      for( std::size_t i = 0; i < N; i++ )
         tuple[ i ] = 1;

      for( std::size_t i = 0; i < prime_factors.size(); i++ ) {
         TNL_ASSERT_LT( components[ i ], 3, "got wrong value from the cartesian power" );
         tuple[ components[ i ] ] *= prime_factors[ i ];
      }

#ifndef NDEBUG
      // verify the result
      Index product = 1;
      for( std::size_t i = 0; i < N; i++ )
         product *= tuple[ i ];
      TNL_ASSERT_EQ( product, number, "integer factorization failed - product is not equal to the input" );
#endif

      result.insert( std::move( tuple ) );
   }

   return result;
}

/**
 * \brief This function swaps bits at positions \e p1 and \e p2 in an integer \e n.
 */
template< typename Index >
Index
swapBits( Index n, std::uint8_t p1, std::uint8_t p2 )
{
   // move the p1-th bit to the rightmost side
   const Index bit1 = ( n >> p1 ) & Index( 1 );

   // move the p2-th to rightmost side
   const Index bit2 = ( n >> p2 ) & Index( 1 );

   // XOR the two bits
   Index x = bit1 ^ bit2;

   // put the XOR-ed bits back to their original positions
   x = ( x << p1 ) | ( x << p2 );

   // XOR `x` with the original number so that the two bits are swapped
   return n ^ x;
}

}  // namespace TNL
