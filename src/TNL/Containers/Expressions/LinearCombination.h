// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Containers/Expressions/ExpressionTemplates.h>

#pragma once

namespace TNL::Containers::Expressions {

template< typename T1, typename T2, typename ValueType >
struct MergeLinearCombinationTypes
{
   using type = decltype( std::declval< T1 >() + std::declval< T2 >() );
};

template< typename T1, typename ValueType >
struct MergeLinearCombinationTypes< T1, ValueType, ValueType >
{
   using type = T1;
};

template< typename T2, typename ValueType >
struct MergeLinearCombinationTypes< ValueType, T2, ValueType >
{
   using type = T2;
};

template< typename ValueType >
struct MergeLinearCombinationTypes< ValueType, ValueType, ValueType >
{
   using type = ValueType;
};

template< typename Coefficients,
          typename Vector,
          int Index,
          int Size = Coefficients::getSize(),
          bool Zero = ( Coefficients::getValue( Index ) == 0 ) >
struct LinearCombinationReturnType
{};

template< typename Coefficients, typename Vector, int Index, int Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, false >
{
   using type = typename MergeLinearCombinationTypes<
      decltype( Coefficients::getValue( Index ) * std::declval< Vector >() ),
      typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type,
      typename Vector::RealType >::type;
};

template< typename Coefficients, typename Vector, int Index, int Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, true >
{
   using type = typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type;
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Index + 1, false >
{
   using type = decltype( Coefficients::getValue( Index ) * std::declval< Vector >() );
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Index + 1, true >
{
   using type = typename Vector::RealType;
};

template< typename Coefficients, typename Vector, int Index, int Size = Coefficients::getSize() >
struct LinearCombinationEvaluation
{
   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   template< typename... OtherVectors >
   static ResultType
   evaluate( const Vector& v, const OtherVectors&... others )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) { // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( Index ) != 0 )
            return Coefficients::getValue( Index ) * v;
         else return 0;
      } else if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * v +
            LinearCombinationEvaluation< Coefficients, Vector, Index + 1, Size >::evaluate( others... );
         else return LinearCombinationEvaluation< Coefficients, Vector, Index + 1, Size >::evaluate( others... );
   }

   static ResultType
   evaluateArray( const Containers::StaticArray< Size, Vector >& vectors )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index + 1 >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) { // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( Index ) != 0 )
            return Coefficients::getValue( Index ) * vectors[ Index ];
         else return 0;
      } else if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * vectors[ Index ] +
            LinearCombinationEvaluation< Coefficients, Vector, Index + 1, Size >::evaluateArray( vectors );
         else return LinearCombinationEvaluation< Coefficients, Vector, Index + 1, Size >::evaluateArray( vectors );
   }
};

template< typename Coefficients, typename Vector, int Index >
struct LinearCombinationEvaluation< Coefficients, Vector, Index, Index + 1 >
{
   static constexpr size_t Size = Coefficients::getSize();

   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   static ResultType
   evaluate( const Vector& v )
   {
      if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * v;
      else
         return 0;
   }

   static ResultType
   evaluateArray( const Containers::StaticArray< Size, Vector >& vectors )
   {
      if constexpr( Coefficients::getValue( Index ) != 0 )
         return Coefficients::getValue( Index ) * vectors[ Index ];
      else
         return 0;
   }
};

template< typename Coefficients, typename Vector >
struct LinearCombination
{
   static constexpr size_t Size = Coefficients::getSize();

   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, 0 >::type;

   template< typename... OtherVectors >
   static ResultType
   evaluate( const OtherVectors&... others )
   {
      static_assert( sizeof...( OtherVectors ) == Size, "Number of input vectors must match number of coefficients" );
      return LinearCombinationEvaluation< Coefficients, Vector, 0, Size >::evaluate( others... );
   }

   static ResultType
   evaluateArray( const Containers::StaticArray< Size, Vector >& vectors )
   {
      return LinearCombinationEvaluation< Coefficients, Vector, 0, Size >::evaluateArray( vectors );
   }
};

}  // namespace TNL::Containers::Expressions
