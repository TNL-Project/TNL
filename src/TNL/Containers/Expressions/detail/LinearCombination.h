// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <type_traits>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>

#pragma once

// std::integral_constant is used due to nvcc. In version 12.2, it does not
// allow  partial specialization with nontype template parameters.

namespace TNL::Containers::Expressions::detail {

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
          typename Index,
          typename Size = std::integral_constant< size_t, Coefficients::getSize() >,
          typename Zero = std::integral_constant< bool, Coefficients::getValue( Index::value ) == 0 > >
struct LinearCombinationReturnType
{};

template< typename Coefficients, typename Vector, typename Index, typename Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, std::integral_constant< bool, false > >
{
   using type = typename MergeLinearCombinationTypes<
      decltype( Coefficients::getValue( Index::value ) * std::declval< Vector >() ),
      typename LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 > >::type,
      typename Vector::RealType >::type;
};

template< typename Coefficients, typename Vector, typename Index, typename Size >
struct LinearCombinationReturnType< Coefficients, Vector, Index, Size, std::integral_constant< bool, true > >
{
   using type = typename LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 > >::type;
};

template< typename Coefficients, typename Vector, typename Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, std::integral_constant< size_t, Index::value + 1 >, std::integral_constant< bool, false > >
{
   using type = decltype( Coefficients::getValue( Index::value ) * std::declval< Vector >() );
};

template< typename Coefficients, typename Vector, typename Index >
struct LinearCombinationReturnType< Coefficients, Vector, Index, std::integral_constant< size_t, Index::value + 1 >, std::integral_constant< bool, true > >
{
   using type = typename Vector::RealType;
};

template< typename Coefficients, typename Vector, typename Index, typename Size = std::integral_constant< size_t, Coefficients::getSize() > >
struct LinearCombinationEvaluation
{
   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   template< typename... OtherVectors >
   static ResultType
   evaluate( const Vector& v, const OtherVectors&... others )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType = typename LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 > >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) { // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( Index::value ) != 0 )
            return Coefficients::getValue( Index::value ) * v;
         else return 0;
      } else if constexpr( Coefficients::getValue( Index::value ) != 0 )
         return Coefficients::getValue( Index::value ) * v +
            LinearCombinationEvaluation< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 >, Size >::evaluate( others... );
         else return LinearCombinationEvaluation< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 >, Size >::evaluate( others... );
   }

   static ResultType
   evaluateArray( const std::array< Vector, Size::value >& vectors )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType = typename LinearCombinationReturnType< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 > >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) { // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( Index::value ) != 0 )
            return Coefficients::getValue( Index::value ) * vectors[ Index::value ];
         else return 0;
      } else if constexpr( Coefficients::getValue( Index::value ) != 0 )
         return Coefficients::getValue( Index::value ) * vectors[ Index::value ] +
            LinearCombinationEvaluation< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 >, Size >::evaluateArray( vectors );
         else return LinearCombinationEvaluation< Coefficients, Vector, std::integral_constant< size_t, Index::value + 1 >, Size >::evaluateArray( vectors );
   }
};

template< typename Coefficients, typename Vector, typename Index >
struct LinearCombinationEvaluation< Coefficients, Vector, Index, std::integral_constant< size_t, Index::value + 1 > >
{
   static constexpr size_t Size = Coefficients::getSize();

   using ResultType = typename LinearCombinationReturnType< Coefficients, Vector, Index >::type;

   static ResultType
   evaluate( const Vector& v )
   {
      if constexpr( Coefficients::getValue( Index::value ) != 0 )
         return Coefficients::getValue( Index::value ) * v;
      else
         return 0;
   }

   static ResultType
   evaluateArray( const std::array< Vector, Size >& vectors )
   {
      if constexpr( Coefficients::getValue( Index::value ) != 0 )
         return Coefficients::getValue( Index::value ) * vectors[ Index::value ];
      else
         return 0;
   }
};

}  // namespace TNL::Containers::Expressions::detail
