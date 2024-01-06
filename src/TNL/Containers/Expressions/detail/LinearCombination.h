// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/ndarray/Meta.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>

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

template< typename... Vectors >
struct VectorTypesWrapper
{
   template< std::size_t i >
   using VectorType = decltype( Containers::detail::get_from_pack< i >( std::declval< Vectors >()... ) );
};

template< typename Vector >
struct ConstantVectorTypesWrapper
{
   template< std::size_t i >
   using VectorType = Vector;
};

template< typename Coefficients,
          typename VectorsWrapper,
          typename CoefficientIndex,
          typename Size = std::integral_constant< std::size_t, Coefficients::getSize() >,
          typename Zero = std::integral_constant< bool, Coefficients::getValue( CoefficientIndex::value ) == 0 > >
struct LinearCombinationReturnType
{};

template< typename Coefficients, typename VectorsWrapper, typename CoefficientIndex, typename Size >
class LinearCombinationReturnType< Coefficients, VectorsWrapper, CoefficientIndex, Size, std::integral_constant< bool, false > >
{
   using CurrentVectorType = typename VectorsWrapper::template VectorType< CoefficientIndex::value >;
   using MultipliedType = decltype( Coefficients::getValue( CoefficientIndex::value ) * std::declval< CurrentVectorType >() );
   using RemainingCombinationType =
      typename LinearCombinationReturnType< Coefficients,
                                            VectorsWrapper,
                                            std::integral_constant< std::size_t, CoefficientIndex::value + 1 > >::type;

public:
   using type = typename MergeLinearCombinationTypes< MultipliedType,
                                                      RemainingCombinationType,
                                                      typename CurrentVectorType::RealType >::type;
};

template< typename Coefficients, typename VectorsWrapper, typename CoefficientIndex, typename Size >
struct LinearCombinationReturnType< Coefficients, VectorsWrapper, CoefficientIndex, Size, std::integral_constant< bool, true > >
{
   using type =
      typename LinearCombinationReturnType< Coefficients,
                                            VectorsWrapper,
                                            std::integral_constant< std::size_t, CoefficientIndex::value + 1 > >::type;
};

template< typename Coefficients, typename VectorsWrapper, typename CoefficientIndex >
class LinearCombinationReturnType< Coefficients,
                                   VectorsWrapper,
                                   CoefficientIndex,
                                   std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                   std::integral_constant< bool, false > >
{
   using CurrentVectorType = typename VectorsWrapper::template VectorType< CoefficientIndex::value >;

public:
   using type = decltype( Coefficients::getValue( CoefficientIndex::value ) * std::declval< CurrentVectorType >() );
};

template< typename Coefficients, typename VectorsWrapper, typename CoefficientIndex >
class LinearCombinationReturnType< Coefficients,
                                   VectorsWrapper,
                                   CoefficientIndex,
                                   std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                   std::integral_constant< bool, true > >
{
   using CurrentVectorType = typename VectorsWrapper::template VectorType< CoefficientIndex::value >;

public:
   using type = typename CurrentVectorType::RealType;
};

template< typename Coefficients,
          typename CoefficientIndex,
          typename Size = std::integral_constant< std::size_t, Coefficients::getSize() > >
struct LinearCombinationEvaluation
{
   template< typename Vector, typename... OtherVectors >
   constexpr static auto
   evaluate( const Vector& v, const OtherVectors&... others )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType =
         typename LinearCombinationReturnType< Coefficients,
                                               VectorTypesWrapper< OtherVectors... >,
                                               std::integral_constant< std::size_t, CoefficientIndex::value + 1 > >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) {  // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
            return Coefficients::getValue( CoefficientIndex::value ) * v;
         else
            return 0;
      }
      else if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
         return Coefficients::getValue( CoefficientIndex::value ) * v
              + LinearCombinationEvaluation< Coefficients,
                                             std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                             Size >::evaluate( others... );
      else
         return LinearCombinationEvaluation< Coefficients,
                                             std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                             Size >::evaluate( others... );
   }

   template< typename Vector >
   constexpr static auto
   evaluate( const std::array< Vector, Size::value >& vectors )
   {
      using ValueType = typename Vector::RealType;
      using AuxResultType =
         typename LinearCombinationReturnType< Coefficients,
                                               ConstantVectorTypesWrapper< Vector >,
                                               std::integral_constant< std::size_t, CoefficientIndex::value + 1 > >::type;
      if constexpr( std::is_same_v< AuxResultType, ValueType > ) {  // the rest of coefficients are zero
         if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
            return Coefficients::getValue( CoefficientIndex::value ) * vectors[ CoefficientIndex::value ];
         else
            return 0;
      }
      else if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
         return Coefficients::getValue( CoefficientIndex::value ) * vectors[ CoefficientIndex::value ]
              + LinearCombinationEvaluation< Coefficients,
                                             std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                             Size >::evaluate( vectors );
      else
         return LinearCombinationEvaluation< Coefficients,
                                             std::integral_constant< std::size_t, CoefficientIndex::value + 1 >,
                                             Size >::evaluate( vectors );
   }
};

template< typename Coefficients, typename CoefficientIndex >
struct LinearCombinationEvaluation< Coefficients,
                                    CoefficientIndex,
                                    std::integral_constant< std::size_t, CoefficientIndex::value + 1 > >
{
   static constexpr std::size_t Size = Coefficients::getSize();

   template< typename Vector >
   constexpr static auto
   evaluate( const Vector& v )
   {
      if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
         return Coefficients::getValue( CoefficientIndex::value ) * v;
      else
         return 0;
   }

   template< typename Vector >
   constexpr static auto
   evaluate( const std::array< Vector, Size >& vectors )
   {
      if constexpr( Coefficients::getValue( CoefficientIndex::value ) != 0 )
         return Coefficients::getValue( CoefficientIndex::value ) * vectors[ CoefficientIndex::value ];
      else
         return 0;
   }
};

}  // namespace TNL::Containers::Expressions::detail
