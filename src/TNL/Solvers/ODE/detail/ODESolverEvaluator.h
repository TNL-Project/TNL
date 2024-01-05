// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/LinearCombination.h>

namespace TNL::Solvers::ODE::detail {

// std::integral_constant is used due to nvcc. In version 12.2, it does not
// allow  partial specialization with nontype template parameters.

template< typename Method, std::size_t Stage >
struct CoefficientsProxy
{
   using ValueType = typename Method::ValueType;

   static constexpr std::size_t
   getSize()
   {
      return Method::getStages();
   }

   static constexpr ValueType
   getValue( std::size_t i )
   {
      return Method::getCoefficient( Stage, i );
   }
};

template< typename Method >
struct UpdateCoefficientsProxy
{
   static constexpr std::size_t
   getSize()
   {
      return Method::getStages();
   }

   using ValueType = typename Method::ValueType;

   static constexpr ValueType
   getValue( std::size_t i )
   {
      return Method::getUpdateCoefficient( i );
   }
};

template< typename Method >
struct ErrorCoefficientsProxy
{
   using ValueType = typename Method::ValueType;

   static constexpr std::size_t
   getSize()
   {
      return Method::getStages();
   }

   static constexpr ValueType
   getValue( std::size_t i )
   {
      return Method::getErrorCoefficient( i );
   }
};

template< typename Method,
          typename Stage = std::integral_constant< std::size_t, 0 >,
          typename Stages = std::integral_constant< std::size_t, Method::getStages() > >
struct StaticODESolverEvaluator
{
   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void __cuda_callable__
   computeKVectors( std::array< VectorView, Stages::value >& k,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    VectorView aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {
      static_assert( IsStaticArrayType< VectorView >::value, "VectorView must be a static array type" );
      if constexpr( Stage::value == 0 ) {  // k[ 0 ] = f( t, u )
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, u, k[ Stage::value ], params... );
         StaticODESolverEvaluator< Method, std::integral_constant< std::size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
      else {
         using Coefficients = CoefficientsProxy< Method, Stage::value >;
         using Formula = Containers::Expressions::LinearCombination< Coefficients, VectorView >;
         aux = u + currentTau * Formula::evaluate( k );
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, aux, k[ Stage::value ], params... );
         StaticODESolverEvaluator< Method, std::integral_constant< std::size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
   }
};

template< typename Method >
struct StaticODESolverEvaluator< Method,
                                 std::integral_constant< std::size_t, Method::getStages() >,
                                 std::integral_constant< std::size_t, Method::getStages() > >
{
   static constexpr std::size_t Stages = Method::getStages();

   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void __cuda_callable__
   computeKVectors( std::array< VectorView, Stages >& k_vectors,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    const VectorView& aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {}
};

/////
// Dynamic ODE solver evaluator
template< typename Method,
          typename Stage = std::integral_constant< std::size_t, 0 >,
          typename Stages = std::integral_constant< std::size_t, Method::getStages() > >
struct ODESolverEvaluator
{
   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void
   computeKVectors( std::array< VectorView, Stages::value >& k,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    VectorView aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {
      static_assert( ! IsStaticArrayType< VectorView >::value, "VectorView must NOT be a static array type" );
      if constexpr( Stage::value == 0 ) {  // k[ 0 ] = f( t, u )
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, u, k[ Stage::value ], params... );
         ODESolverEvaluator< Method, std::integral_constant< std::size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
      else {
         using Coefficients = CoefficientsProxy< Method, Stage::value >;
         using Formula = Containers::Expressions::LinearCombination< Coefficients, VectorView >;
         aux = u + currentTau * Formula::evaluate( k );
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, aux, k[ Stage::value ], params... );
         ODESolverEvaluator< Method, std::integral_constant< std::size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
   }
};

template< typename Method >
struct ODESolverEvaluator< Method,
                           std::integral_constant< std::size_t, Method::getStages() >,
                           std::integral_constant< std::size_t, Method::getStages() > >
{
   static constexpr std::size_t Stages = Method::getStages();

   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void
   computeKVectors( std::array< VectorView, Stages >& k_vectors,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    const VectorView& aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {}
};

}  // namespace TNL::Solvers::ODE::detail
