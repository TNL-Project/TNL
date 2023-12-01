// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <cstddef>

namespace TNL::Solvers::ODE::detail {

// std::integral_constant is used due to nvcc. In version 12.2, it does not
// allow  partial specialization with nontype template parameters.

template< typename Method, size_t Stage >
struct CoefficientsProxy
{
   using ValueType = typename Method::ValueType;

   static constexpr size_t
   getSize()
   {
      return Method::getStages();
   }

   static constexpr ValueType
   getValue( size_t i )
   {
      return Method::getCoefficient( Stage, i );
   }
};

template< typename Method >
struct UpdateCoefficientsProxy
{
   static constexpr size_t
   getSize()
   {
      return Method::getStages();
   }

   using ValueType = typename Method::ValueType;

   static constexpr ValueType
   getValue( size_t i )
   {
      return Method::getUpdateCoefficient( i );
   }
};

template< typename Method >
struct ErrorCoefficientsProxy
{
   using ValueType = typename Method::ValueType;

   static constexpr size_t
   getSize()
   {
      return Method::getStages();
   }

   static constexpr ValueType
   getValue( size_t i )
   {
      return Method::getErrorCoefficient( i );
   }
};

template< typename Method,
          typename Stage = std::integral_constant< size_t, 0 >,
          typename Stages = std::integral_constant< size_t, Method::getStages() > >
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
      if constexpr( Stage::value == 0 ) {  // k[ 0 ] = f( t, u )
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, u, k[ Stage::value ], params... );
         ODESolverEvaluator< Method, std::integral_constant< size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
      else {
         using Coefficients = CoefficientsProxy< Method, Stage::value >;
         using Formula = Containers::Expressions::LinearCombination< Coefficients, VectorView >;
         aux = u + currentTau * Formula::evaluateArray( k );
         rhsFunction(
            time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, aux, k[ Stage::value ], params... );
         ODESolverEvaluator< Method, std::integral_constant< size_t, Stage::value + 1 > >::computeKVectors(
            k, time, currentTau, u, aux, rhsFunction, params... );
      }
   }
};

template< typename Method >
struct ODESolverEvaluator< Method,
                           std::integral_constant< size_t, Method::getStages() >,
                           std::integral_constant< size_t, Method::getStages() > >
{
   static constexpr size_t Stages = Method::getStages();

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
