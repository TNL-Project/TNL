// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <type_traits>
#include <vector>

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
   return ( ( x & ( x - 1 ) ) == 0 );
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
   return ( ( x & ( x - 1 ) ) == 0 );
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
 * \brief Computes the sum of all i-combinations in set of n elements for i from 0 up to k-1.
 *
 * \tparam Index is an integral type used for the evaluation of the sum.
 * \param k denotes the number of terms in the sum.
 * \param n denotes the number of elements in the set.
 * \return Sum of all i-combinations in set of n elements for i from 0 up to k-1.
 */
template< typename Index >
constexpr Index
firstKCombinationsSum( Index k, Index n )
{
   if( k == 0 )
      return 0;

   if( k == n )
      return ( 1 << n ) - 1;

   if( k == n + 1 )
      return 1 << n;

   Index result = 0;
   for( Index i = 0; i < k; i++ )
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

}  // namespace TNL
