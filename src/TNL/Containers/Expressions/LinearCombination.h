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
 * \brief Linear combination of vectors.
 *
 * This class creates an expression template for a linear combination of vectors i.e.
 *
 * \f[
 *  \vec ET = \alpha_1 \vec v_1 + \alpha_2 \vec v_2 + \dots + \alpha_n \vec v_n
 * \f]
 *
 * The coefficients \f$ \alpha_i \f$ are given as a Coefficients object, which must provide a
 * static method `getValue( int i )` that returns the i-th coefficient. The vectors are given
 * as arguments to the \c evaluate() method. The vectors can by given in a form of
 * \ref std::array or as a parameter pack. The transformation to the expression template is done
 * at compile time and so the coefficients must be static.
 *
 * \tparam Coefficients object with static parameters of the linear combination. It has to implement
 *    static method `getValue( int i )` that returns the i-th coefficient.
 * \tparam Vector is type of the vectors in the linear combination. Can be any type supporrted by the
 *    expression templates i.e. \ref TNL::Containers::StaticVector, \ref TNL::Containers::Vector or
 *    \ref TNL::Containers::DistributedVector.
 */
template< typename Coefficients, typename Vector, bool isStatic = IsStaticArrayType< Vector >::value >
struct LinearCombination;

template< typename Coefficients, typename Vector >
struct LinearCombination< Coefficients, Vector, false >
{
   static constexpr std::size_t
   getSize()
   {
      return Coefficients::getSize();
   }

   using ResultType =
      typename detail::LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< std::size_t, 0 > >::type;

   /**
    * \brief Evaluate the linear combination for vectors given as a parameter pack.
    *
    * \tparam OtherVectors type of parameter pack.
    * \param others input vectors.
    * \return expression template representing the linear combination.
    */
   template< typename... OtherVectors >
   static ResultType
   evaluate( const OtherVectors&... others )
   {
      static_assert( sizeof...( OtherVectors ) == getSize(), "Number of input vectors must match number of coefficients" );
      return detail::LinearCombinationEvaluation< Coefficients,
                                                  Vector,
                                                  std::integral_constant< std::size_t, 0 >,
                                                  std::integral_constant< std::size_t, getSize() > >::evaluate( others... );
   }

   /**
    * \brief Evaluate the linear combination for vectors given by a static array.
    *
    * \param vectors is an array with input vectors.
    * \return expression template representing the linear combination.
    */
   static ResultType
   evaluate( const std::array< Vector, getSize() >& vectors )
   {
      return detail::LinearCombinationEvaluation< Coefficients,
                                                  Vector,
                                                  std::integral_constant< std::size_t, 0 >,
                                                  std::integral_constant< std::size_t, getSize() > >::evaluateArray( vectors );
   }
};

template< typename Coefficients, typename Vector >
struct LinearCombination< Coefficients, Vector, true >
{
   static constexpr std::size_t
   getSize()
   {
      return Coefficients::getSize();
   }

   using ResultType =
      typename detail::LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< std::size_t, 0 > >::type;

   /**
    * \brief Evaluate the linear combination for vectors given as a parameter pack.
    *
    * \tparam OtherVectors type of parameter pack.
    * \param others input vectors.
    * \return expression template representing the linear combination.
    */
   template< typename... OtherVectors >
   __cuda_callable__
   static ResultType
   evaluate( const OtherVectors&... others )
   {
      static_assert( sizeof...( OtherVectors ) == getSize(), "Number of input vectors must match number of coefficients" );
      return detail::LinearCombinationEvaluation< Coefficients,
                                                  Vector,
                                                  std::integral_constant< std::size_t, 0 >,
                                                  std::integral_constant< std::size_t, getSize() > >::evaluate( others... );
   }

   /**
    * \brief Evaluate the linear combination for vectors given by a static array.
    *
    * \param vectors is an array with input vectors.
    * \return expression template representing the linear combination.
    */
   __cuda_callable__
   static ResultType
   evaluate( const std::array< Vector, getSize() >& vectors )
   {
      return detail::LinearCombinationEvaluation< Coefficients,
                                                  Vector,
                                                  std::integral_constant< std::size_t, 0 >,
                                                  std::integral_constant< std::size_t, getSize() > >::evaluateArray( vectors );
   }
};

}  // namespace TNL::Containers::Expressions
