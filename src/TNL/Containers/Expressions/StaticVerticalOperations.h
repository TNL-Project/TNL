// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <TNL/Containers/Expressions/TypeTraits.h>

////
// By vertical operations we mean those applied across vector elements or
// vector expression elements. It means for example minim/maximum of all
// vector elements etc.
namespace TNL::Containers::Expressions {

template< typename Expression >
constexpr auto
StaticExpressionMin( const Expression& expression )
{
   // use argument-dependent lookup and make TNL::min available for unqualified calls
   using TNL::min;
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = min( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
constexpr auto
StaticExpressionArgMin( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   int arg = 0;
   ResultType value = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ ) {
      if( expression[ i ] < value ) {
         value = expression[ i ];
         arg = i;
      }
   }
   return std::make_pair( value, arg );
}

template< typename Expression >
constexpr auto
StaticExpressionMax( const Expression& expression )
{
   // use argument-dependent lookup and make TNL::max available for unqualified calls
   using TNL::max;
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = max( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
constexpr auto
StaticExpressionArgMax( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   int arg = 0;
   ResultType value = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ ) {
      if( expression[ i ] > value ) {
         value = expression[ i ];
         arg = i;
      }
   }
   return std::make_pair( value, arg );
}

template< typename Expression >
constexpr auto
StaticExpressionSum( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux += expression[ i ];
   return aux;
}

template< typename Expression >
constexpr auto
StaticExpressionProduct( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux *= expression[ i ];
   return aux;
}

}  // namespace TNL::Containers::Expressions
