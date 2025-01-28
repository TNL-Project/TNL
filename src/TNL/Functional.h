// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <limits>

#include <TNL/Math.h>

namespace TNL {

/**
 * \defgroup ReductionFunctionObjects Function objects for reduction operations
 *
 * Reduction function objects are used in reduction operations in TNL, such as
 * \ref TNL::Algorithms::reduce but also in reduction in \ref TNL::Algorithms::Segments and
 * \ref TNL::Matrices.
 *
 * In general, each reduction operation is implemented as a generic `operator()` of the form
 * `auto operator()( T1 x, T2 y ) const -> T3`, which computes an appropriate *associative* and
 * *commutative* binary operation. Furthermore, the reduction function objects in TNL *extend* the corresponding
 * [STL function objects](https://en.cppreference.com/w/cpp/utility/functional#Operator_function_objects)
 * with a `getIdentity()` method, which returns the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 * for the operation. The identity element is a value that does not change the result of the operation when combined with any
 * other value.
 */

/**
 * \brief Function object implementing `x + y`.
 * \ingroup ReductionFunctionObjects
 */
struct Plus : public std::plus< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return 0;
   }
};

/**
 * \brief Function object implementing `x - y`.
 */
using Minus = std::minus< void >;

/**
 * \brief Function object implementing `x * y`.
 * \ingroup ReductionFunctionObjects
 */
struct Multiplies : public std::multiplies< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return 1;
   }
};

/**
 * \brief Function object implementing `x / y`.
 */
using Divides = std::divides< void >;

/**
 * \brief Function object implementing `x % y`.
 */
using Modulus = std::modulus< void >;

/**
 * \brief Function object implementing `+x`.
 */
struct UnaryPlus
{
   template< typename T >
   constexpr auto
   operator()( const T& x ) const -> decltype( +x )
   {
      return +x;
   }
};

/**
 * \brief Function object implementing `-x`.
 */
using UnaryMinus = std::negate< void >;

/**
 * \brief Function object implementing `x && y`.
 * \ingroup ReductionFunctionObjects
 */
struct LogicalAnd : public std::logical_and< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }
};

/**
 * \brief Function object implementing `x || y`.
 * \ingroup ReductionFunctionObjects
 */
struct LogicalOr : public std::logical_or< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return 0;
   }
};

/**
 * \brief Function object implementing `!x`.
 */
using LogicalNot = std::logical_not< void >;

/**
 * \brief Extension of \ref std::bit_and<void> for use with \ref TNL::Algorithms::reduce.
 * \ingroup ReductionFunctionObjects
 */
struct BitAnd : public std::bit_and< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return ~static_cast< T >( 0 );
   }
};

/**
 * \brief Extension of \ref std::bit_or<void> for use with \ref TNL::Algorithms::reduce.
 * \ingroup ReductionFunctionObjects
 */
struct BitOr : public std::bit_or< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return 0;
   }
};

/**
 * \brief Extension of \ref std::bit_xor<void> for use with \ref TNL::Algorithms::reduce.
 * \ingroup ReductionFunctionObjects
 */
struct BitXor : public std::bit_xor< void >
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return 0;
   }
};

/**
 * \brief Function object implementing `~x`.
 */
using BitNot = std::bit_not< void >;

/**
 * \brief Function object implementing `x == y`.
 */
using EqualTo = std::equal_to< void >;

/**
 * \brief Function object implementing `x != y`.
 */
using NotEqualTo = std::not_equal_to< void >;

/**
 * \brief Function object implementing `x > y`.
 */
using Greater = std::greater< void >;

/**
 * \brief Function object implementing `x < y`.
 */
using Less = std::less< void >;

/**
 * \brief Function object implementing `x >= y`.
 */
using GreaterEqual = std::greater_equal< void >;

/**
 * \brief Function object implementing `x <= y`.
 */
using LessEqual = std::less_equal< void >;

/**
 * \brief Function object implementing `min(x, y)`.
 * \ingroup ReductionFunctionObjects
 */
struct Min
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }

   template< typename T1, typename T2 >
   constexpr auto
   operator()( const T1& lhs, const T2& rhs ) const
   {
      // use argument-dependent lookup and make TNL::min available for unqualified calls
      using TNL::min;
      return min( lhs, rhs );
   }
};

/**
 * \brief Function object implementing `max(x, y)`.
 * \ingroup ReductionFunctionObjects
 */
struct Max
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::lowest();
   }

   template< typename T1, typename T2 >
   constexpr auto
   operator()( const T1& lhs, const T2& rhs ) const
   {
      // use argument-dependent lookup and make TNL::max available for unqualified calls
      using TNL::max;
      return max( lhs, rhs );
   }
};

/**
 * \defgroup ReductionFunctionObjectsWithArgument Function objects for reduction operations with argument
 *
 * Function objects in this group implement a similar concept as the function objects in the
 * \ref ReductionFunctionObjects group. They are intended for *reductions with argument*,
 * such as the \ref TNL::Algorithms::reduceWithArgument function. Here the reduction operation
 * works not only with *values*, but also *indexes* of the values, which can be used to compute
 * e.g. the position of a minimum or maximum in a sequence of values.
 *
 * In general, each reduction operation with argument is implemented as a generic `operator()`
 * of the form `auto operator()( Value& x, Value y, Index& i, Index j ) const -> void`, which
 * computes an appropriate *associative* and *commutative* binary operation on the `x` and `y`
 * values, updates the result into the `x`, and also updates the index `i` as appropriate.
 *
 * The function objects in this group also implement the `getIdentity()` method, which returns the
 * [identity element](https://en.wikipedia.org/wiki/Identity_element) for the operation in the same
 * way as function objects in the \ref ReductionFunctionObjects group.
 */

/**
 * \brief Function object implementing `argmin(x, y, i, j)`, i.e. returning the minimum value and its index.
 * \ingroup ReductionFunctionObjectsWithArgument
 */
struct MinWithArg
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }

   template< typename Value, typename Index >
   constexpr void
   operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx ) const
   {
      if( lhs > rhs ) {
         lhs = rhs;
         lhsIdx = rhsIdx;
      }
      else if( lhs == rhs && rhsIdx < lhsIdx ) {
         lhsIdx = rhsIdx;
      }
   }
};

/**
 * \brief Function object implementing `argmax(x, y, i, j)`, i.e. returning the maximum value and its index.
 * \ingroup ReductionFunctionObjectsWithArgument
 */
struct MaxWithArg
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::lowest();
   }

   template< typename Value, typename Index >
   constexpr void
   operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx ) const
   {
      if( lhs < rhs ) {
         lhs = rhs;
         lhsIdx = rhsIdx;
      }
      else if( lhs == rhs && rhsIdx < lhsIdx ) {
         lhsIdx = rhsIdx;
      }
   }
};

/**
 * \brief Function object implementing `argany(x, y, i, j)`, i.e. returning the first `true` value and its index.
 * \ingroup ReductionFunctionObjectsWithArgument
 */
struct AnyWithArg
{
   /**
    * \brief Returns the [identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation.
    *
    * Suitable for \ref TNL::Algorithms::reduce.
    */
   template< typename T >
   static constexpr T
   getIdentity()
   {
      return false;
   }

   template< typename Index >
   constexpr void
   operator()( bool& lhs, const bool& rhs, Index& lhsIdx, const Index& rhsIdx ) const
   {
      if( ! lhs && rhs ) {
         lhs = true;
         lhsIdx = rhsIdx;
      }
      else if( lhs && rhs && rhsIdx < lhsIdx )
         lhsIdx = rhsIdx;
   }
};

#define TNL_MAKE_UNARY_FUNCTIONAL( name, function )               \
   struct name                                                    \
   {                                                              \
      template< typename T >                                      \
      __cuda_callable__                                           \
      auto                                                        \
      operator()( const T& x ) const -> decltype( function( x ) ) \
      {                                                           \
         return function( x );                                    \
      }                                                           \
   };

#define TNL_MAKE_BINARY_FUNCTIONAL( name, function )                               \
   struct name                                                                     \
   {                                                                               \
      template< typename T1, typename T2 >                                         \
      __cuda_callable__                                                            \
      auto                                                                         \
      operator()( const T1& x, const T2& y ) const -> decltype( function( x, y ) ) \
      {                                                                            \
         return function( x, y );                                                  \
      }                                                                            \
   };

TNL_MAKE_UNARY_FUNCTIONAL( Abs, abs )
TNL_MAKE_UNARY_FUNCTIONAL( Exp, exp )
TNL_MAKE_UNARY_FUNCTIONAL( Sqr, sqr )
TNL_MAKE_UNARY_FUNCTIONAL( Sqrt, sqrt )
TNL_MAKE_UNARY_FUNCTIONAL( Cbrt, cbrt )
TNL_MAKE_UNARY_FUNCTIONAL( Log, log )
TNL_MAKE_UNARY_FUNCTIONAL( Log10, log10 )
TNL_MAKE_UNARY_FUNCTIONAL( Log2, log2 )
TNL_MAKE_UNARY_FUNCTIONAL( Sin, sin )
TNL_MAKE_UNARY_FUNCTIONAL( Cos, cos )
TNL_MAKE_UNARY_FUNCTIONAL( Tan, tan )
TNL_MAKE_UNARY_FUNCTIONAL( Asin, asin )
TNL_MAKE_UNARY_FUNCTIONAL( Acos, acos )
TNL_MAKE_UNARY_FUNCTIONAL( Atan, atan )
TNL_MAKE_UNARY_FUNCTIONAL( Sinh, sinh )
TNL_MAKE_UNARY_FUNCTIONAL( Cosh, cosh )
TNL_MAKE_UNARY_FUNCTIONAL( Tanh, tanh )
TNL_MAKE_UNARY_FUNCTIONAL( Asinh, asinh )
TNL_MAKE_UNARY_FUNCTIONAL( Acosh, acosh )
TNL_MAKE_UNARY_FUNCTIONAL( Atanh, atanh )
TNL_MAKE_UNARY_FUNCTIONAL( Floor, floor )
TNL_MAKE_UNARY_FUNCTIONAL( Ceil, ceil )
TNL_MAKE_UNARY_FUNCTIONAL( Sign, sign )

TNL_MAKE_BINARY_FUNCTIONAL( Pow, pow )

#undef TNL_MAKE_UNARY_FUNCTIONAL
#undef TNL_MAKE_BINARY_FUNCTIONAL

template< typename ResultType >
struct Cast
{
   struct Operation
   {
      template< typename T >
      constexpr auto
      operator()( const T& a ) const -> ResultType
      {
         return static_cast< ResultType >( a );
      }
   };
};

}  // namespace TNL
