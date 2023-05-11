// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>

namespace TNL::Containers {

/**
 * \brief Vector with constant size.
 *
 * \param Size Size of static vector. Number of its elements.
 * \param Real Type of the values in the static vector.
 */
template< int Size, typename Real = double >
class StaticVector : public StaticArray< Size, Real >
{
public:
   /**
    * \brief Type of numbers stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Indexing type
    */
   using IndexType = int;

   /**
    * \brief Default constructor.
    */
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticVector() = default;

   /**
    * \brief Default copy constructor.
    */
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticVector( const StaticVector& ) = default;

   /**
    * \brief Default copy-assignment operator.
    */
   constexpr StaticVector&
   operator=( const StaticVector& ) = default;

   /**
    * \brief Default move-assignment operator.
    */
   constexpr StaticVector&
   operator=( StaticVector&& ) noexcept = default;

   //! Constructors and assignment operators are inherited from the class \ref StaticArray.
   using StaticArray< Size, Real >::StaticArray;
#if ! defined( __CUDACC_VER_MAJOR__ ) || __CUDACC_VER_MAJOR__ < 11
   using StaticArray< Size, Real >::operator=;
#endif

   /**
    * \brief Constructor from binary vector expression.
    *
    * \param expr is binary expression.
    */
   template< typename T1, typename T2, typename Operation >
   constexpr StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expr );

   /**
    * \brief Constructor from unary expression.
    *
    * \param expr is unary expression
    */
   template< typename T, typename Operation >
   constexpr StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& expr );

   /**
    * \brief Assignment operator with a vector expression.
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator=( const VectorExpression& expression );

   /**
    * \brief Addition operator with a vector expression
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator+=( const VectorExpression& expression );

   /**
    * \brief Subtraction operator with a vector expression.
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator-=( const VectorExpression& expression );

   /**
    * \brief Elementwise multiplication by a vector expression.
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression.
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator*=( const VectorExpression& expression );

   /**
    * \brief Elementwise division by a vector expression.
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator/=( const VectorExpression& expression );

   /**
    * \brief Elementwise modulo by a vector expression.
    *
    * The vector expression can be even just static vector.
    *
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   constexpr StaticVector&
   operator%=( const VectorExpression& expression );
};

// Enable expression templates for StaticVector
namespace Expressions {
template< int Size, typename Real >
struct HasEnabledStaticExpressionTemplates< StaticVector< Size, Real > > : std::true_type
{};
}  // namespace Expressions

}  // namespace TNL::Containers

// specializations to make StaticVector work with C++17 structured bindings
// (all these specializations exist for std::array)
namespace std {

template< int N, class T >
struct tuple_size< TNL::Containers::StaticVector< N, T > > : std::integral_constant< std::size_t, N >
{};

template< std::size_t I, int N, class T >
struct tuple_element< I, TNL::Containers::StaticVector< N, T > >
{
   using type = T;
};

}  // namespace std

// the `get` function must be defined in the TNL::Containers namespace,
// because structured binding finds it by ADL
namespace TNL::Containers {

template< std::size_t I, int N, class T >
constexpr T&
get( StaticVector< N, T >& a ) noexcept
{
   static_assert( I < N );
   return a[ I ];
}

template< std::size_t I, int N, class T >
constexpr T&&
get( StaticVector< N, T >&& a ) noexcept
{
   static_assert( I < N );
   return std::move( a[ I ] );
}

template< std::size_t I, int N, class T >
constexpr const T&
get( const StaticVector< N, T >& a ) noexcept
{
   static_assert( I < N );
   return a[ I ];
}

template< std::size_t I, int N, class T >
constexpr const T&&
get( const StaticVector< N, T >&& a ) noexcept
{
   static_assert( I < N );
   return std::move( a[ I ] );
}

}  // namespace TNL::Containers

#include <TNL/Containers/StaticVector.hpp>

// TODO: move to some other source file
namespace TNL::Containers {

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >
VectorProduct( const StaticVector< 3, Real >& u, const StaticVector< 3, Real >& v )
{
   StaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
}

template< typename Real >
__cuda_callable__
Real
TriangleArea( const StaticVector< 2, Real >& a, const StaticVector< 2, Real >& b, const StaticVector< 2, Real >& c )
{
   StaticVector< 3, Real > u1;
   StaticVector< 3, Real > u2;
   u1.x() = b.x() - a.x();
   u1.y() = b.y() - a.y();
   u1.z() = 0.0;
   u2.x() = c.x() - a.x();
   u2.y() = c.y() - a.y();
   u2.z() = 0;

   const StaticVector< 3, Real > v = VectorProduct( u1, u2 );
   return 0.5 * TNL::sqrt( dot( v, v ) );
}

template< typename Real >
__cuda_callable__
Real
TriangleArea( const StaticVector< 3, Real >& a, const StaticVector< 3, Real >& b, const StaticVector< 3, Real >& c )
{
   StaticVector< 3, Real > u1;
   StaticVector< 3, Real > u2;
   u1.x() = b.x() - a.x();
   u1.y() = b.y() - a.y();
   u1.z() = b.z() - a.z();
   u2.x() = c.x() - a.x();
   u2.y() = c.y() - a.y();
   u2.z() = c.z() - a.z();

   const StaticVector< 3, Real > v = VectorProduct( u1, u2 );
   return 0.5 * TNL::sqrt( dot( v, v ) );
}

}  // namespace TNL::Containers
