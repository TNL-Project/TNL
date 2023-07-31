// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Expressions/TypeTraits.h>

namespace TNL::Containers::Expressions {

enum ExpressionVariableType
{
   ArithmeticVariable,
   VectorExpressionVariable,
   OtherVariable
};

template< typename T, typename V = T >
constexpr ExpressionVariableType
getExpressionVariableType()
{
   if constexpr( IsScalarType< T >::value || is_complex_v< T > )
      return ArithmeticVariable;
   // vectors must be considered as an arithmetic type when used as RealType in another vector
   else if constexpr( IsArithmeticSubtype< T, V >::value )
      return ArithmeticVariable;
   else if constexpr( HasEnabledExpressionTemplates< T >::value || HasEnabledStaticExpressionTemplates< T >::value
                      || HasEnabledDistributedExpressionTemplates< T >::value )
      return VectorExpressionVariable;
   else if constexpr( IsArrayType< T >::value || IsStaticArrayType< T >::value )
      return VectorExpressionVariable;
   else
      return OtherVariable;
}

}  // namespace TNL::Containers::Expressions
