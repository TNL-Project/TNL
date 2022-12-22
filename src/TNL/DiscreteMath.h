// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
namespace TNL {

/**
 * \brief Computes power of a value in discrete numbers.
 *
 * \param value is the argument of the power function.
 * \param power is the exponent in the power function.
 * \return constexpr size_t is the result of the power function.
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
 * \brief Computes product of numbers first * (first+1) * ... * last.
 *
 * \tparam Index is a discrete type used for evaluation of the product.
 * \param first is the first number of the product
 * \param last is the last number of the product.
 * \return constexpr Index is result of the product.
 */
template< typename Index >
constexpr Index
product( Index begin, Index end )
{
   Index result = 1;
   for( Index i = begin; i <= end; i++ )
      result *= i;
   return result;
}

/**
 * \brief Computes number of k-combinations in set of n element.
 *
 * See [Wikipedie](https://en.wikipedia.org/wiki/Combination)
 *
 * \tparam Index is a discrete type used for evaluation of the number of combinations.
 * \param k denotes number of elements in one combination.
 * \param n denotes number of all elements in the set.
 * \return constexpr Index is number of all k-combinations in set of n elements.
 */
template< typename Index >
constexpr Index
combinationsCount( Index k, Index n )
{
   if( k < 0 )
      return 0;
   return product< Index >( k + 1, n ) / product< Index >( 1, n - k );
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

} // namespace TNL
