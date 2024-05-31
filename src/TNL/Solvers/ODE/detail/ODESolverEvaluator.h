// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/LinearCombination.h>
#include <TNL/Algorithms/staticFor.h>

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

template< typename Method >
struct StaticODESolverEvaluator
{
   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void __cuda_callable__
   computeKVectors( std::array< VectorView, Method::getStages() >& k,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    VectorView aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {
      static_assert( IsStaticArrayType< VectorView >::value, "VectorView must be a static array type" );

      rhsFunction( time + Method::getTimeCoefficient( 0 ) * currentTau, currentTau, u, k[ 0 ], params... );
      Algorithms::staticFor< int, 1, Method::getStages() >(
         [ & ]( auto stage )
         {
            using Coefficients = CoefficientsProxy< Method, stage >;
            aux = u + currentTau * Containers::linearCombination< Coefficients >( k );
            rhsFunction( time + Method::getTimeCoefficient( stage ) * currentTau, currentTau, aux, k[ stage ], params... );
         } );
   }
};

// Dynamic ODE solver evaluator
template< typename Method >
struct ODESolverEvaluator
{
   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction, typename... Params >
   static void
   computeKVectors( std::array< VectorView, Method::getStages() >& k,
                    const Value& time,
                    const Value& currentTau,
                    const ConstVectorView& u,
                    VectorView aux,
                    RHSFunction&& rhsFunction,
                    Params&&... params )
   {
      static_assert( ! IsStaticArrayType< VectorView >::value, "VectorView must NOT be a static array type" );

      rhsFunction( time + Method::getTimeCoefficient( 0 ) * currentTau, currentTau, u, k[ 0 ], params... );
      Algorithms::staticFor< int, 1, Method::getStages() >(
         [ & ]( auto stage )
         {
            using Coefficients = CoefficientsProxy< Method, stage >;
            aux = u + currentTau * Containers::linearCombination< Coefficients >( k );
            rhsFunction( time + Method::getTimeCoefficient( stage ) * currentTau, currentTau, aux, k[ stage ], params... );
         } );
   }
};

}  // namespace TNL::Solvers::ODE::detail
