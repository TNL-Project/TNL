// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Containers/Expressions/ExpressionTemplates.h>

#pragma once

namespace TNL::Containers::Expressions {

template< typename T1, typename T2 >
struct MergeLinearCombinationTypes
{
   using type = decltype( std::declval< T1 >() + std::declval< T2 >() );
};

template< typename T1 >
struct MergeLinearCombinationTypes< T1, void >
{
   using type = T1;
};

template< typename T2 >
struct MergeLinearCombinationTypes< void, T2 >
{
   using type = T2;
};

template<>
struct MergeLinearCombinationTypes< void, void >
{
   using type = void;
};

template< typename Coefficients,
          typename Vector,
          int Index,
          int Size = Coefficients::getSize(),
          bool Nonzero = ( Coefficients::getValue( Index ) != 0 ) >
struct LinearCombinationReturnType
{};

template< typename Coefficients, typename Vector, int Index, int Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, true >
{
   using type = typename MergeLinearCombinationTypes<
      decltype( Coefficients::getValue( Index ) * std::declval< Vector >() ),
      typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type >::type;
};

template< typename Coefficients, typename Vector, int Index, int Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, false >
{
   using type = typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type;
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Index + 1, true >
{
   using type = decltype( Coefficients::getValue( Index ) * std::declval< Vector >() );
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Index + 1, false >
{
   using type = typename Vector::RealType;
};

template< typename Coefficients, typename Vector, int Index, int Size = Coefficients::getSize() >
struct LinearCombinationEvaluation
{
   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   template< typename OutVector, typename InVector, typename... InVectors >
   static void
   evaluate( OutVector& out, const InVector& in, const InVectors&... rest )
   {
      /*if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * in +
            LinearCombinationEvaluation< Coefficients, Index + 1 >::evaluate( rest... );
      else return  LinearCombinationEvaluation< Coefficients, Index + 1 >::evaluate( rest... );*/
   }
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationEvaluation< Coefficients, Vector, Index, Index + 1 >
{
   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   template< typename... OtherVectors >
   static ResultType
   evaluate( const Vector& in, const OtherVectors&... rest )
   {
      if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * in;
      else
         return 0;
   }
};

template< typename Coefficients, typename Vector >
struct LinearCombination
{
   static constexpr size_t size = Coefficients::getSize();

   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, 0 >::type;

   template< typename... OtherVectors >
   static ResultType
   evaluate( const Vector& v, const OtherVectors&... in )
   {
      static_assert( sizeof...( OtherVectors ) == size, "Number of input vectors must match number of coefficients" );
      return LinearCombinationEvaluation< Coefficients, Vector, 0 >::evaluate( v, in... );
   }
};

}  // namespace TNL::Containers::Expressions
