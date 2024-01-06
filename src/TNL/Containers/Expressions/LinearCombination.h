// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "detail/LinearCombination.h"

namespace TNL::Containers::Expressions {

/**
 * \brief Generates an expression for a linear combination of vectors.
 *
 * This function creates an expression for a linear combination of vectors, i.e.
 *
 * \f[
 *    \vec ET = \alpha_1 \vec v_1 + \alpha_2 \vec v_2 + \dots + \alpha_n \vec v_n
 * \f]
 *
 * The coefficients \f$ \alpha_i \f$ are given as a `Coefficients` class, which must provide a
 * static method `getValue( int i )` that returns the i-th coefficient. The vectors are given
 * as arguments to the function. The transformation to the expression template is done at compile
 * time and so the coefficients must be static.
 *
 * \tparam Coefficients Class with static parameters of the linear combination. It must provide a
 *    static method `getValue( int i )` that returns the i-th coefficient.
 * \tparam Vector Type of the vectors in the linear combination. It can be any vector type that
 *    can be combined with expression templates, i.e. \ref TNL::Containers::StaticVector,
 *    \ref TNL::Containers::Vector or \ref TNL::Containers::DistributedVector.
 * \param vectors Input vectors that will be combined with `Coefficients`.
 * \returns An expression object representing the linear combination.
 */
template< class Coefficients, typename Vector >
constexpr auto
linearCombination( const std::array< Vector, Coefficients::getSize() >& vectors ) ->
   typename detail::LinearCombinationReturnType< Coefficients,
                                                 detail::ConstantVectorTypesWrapper< Vector >,
                                                 std::integral_constant< std::size_t, 0 > >::type
{
   return detail::LinearCombinationEvaluation<
      Coefficients,
      std::integral_constant< std::size_t, 0 >,
      std::integral_constant< std::size_t, Coefficients::getSize() > >::evaluate( vectors );
}

/**
 * \brief Generates an expression for a linear combination of vectors.
 *
 * This function creates an expression for a linear combination of vectors, i.e.
 *
 * \f[
 *    \vec ET = \alpha_1 \vec v_1 + \alpha_2 \vec v_2 + \dots + \alpha_n \vec v_n
 * \f]
 *
 * The coefficients \f$ \alpha_i \f$ are given as a `Coefficients` class, which must provide a
 * static method `getValue( int i )` that returns the i-th coefficient. The vectors are given
 * as arguments to the function. The transformation to the expression template is done at compile
 * time and so the coefficients must be static.
 *
 * \tparam Coefficients Class with static parameters of the linear combination. It must provide a
 *    static method `getValue( int i )` that returns the i-th coefficient.
 * \tparam Vectors A variadic pack of the vector types in the linear combination. Each pack can
 *    contain any vector types that can be combined with expression templates, i.e.
 *    \ref TNL::Containers::StaticVector, \ref TNL::Containers::Vector or
 *    \ref TNL::Containers::DistributedVector.
 * \param vectors Input vectors that will be combined with `Coefficients`.
 * \returns An expression object representing the linear combination.
 */
template<
   class Coefficients,
   typename... Vectors,
   std::enable_if_t< IsArrayType< decltype( Containers::detail::get_from_pack< 0 >( std::declval< Vectors >()... ) ) >::value,
                     bool > = true >
constexpr auto
linearCombination( const Vectors&... vectors ) ->
   typename detail::LinearCombinationReturnType< Coefficients,
                                                 detail::VectorTypesWrapper< Vectors... >,
                                                 std::integral_constant< std::size_t, 0 > >::type
{
   static_assert( sizeof...( Vectors ) == Coefficients::getSize(),
                  "Number of input vectors must match number of coefficients" );
   return detail::LinearCombinationEvaluation<
      Coefficients,
      std::integral_constant< std::size_t, 0 >,
      std::integral_constant< std::size_t, Coefficients::getSize() > >::evaluate( vectors... );
}

}  // namespace TNL::Containers::Expressions
