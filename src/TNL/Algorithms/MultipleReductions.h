// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <TNL/Backend/Macros.h>

namespace TNL::Algorithms {

/**
 * \brief A tuple of the values of several reductions computed at once by \ref MultipleReductions.
 *
 * Unlike `std::tuple`, it is an aggregate of its components, so it is trivially copyable when all
 * components are trivially copyable. Hence it is reduced by the warp shuffles in the reduction
 * kernels on GPUs and it can be used in the device code. The components are accessed by
 * `get< I >()` or by structured bindings:
 *
 * ```cpp
 * auto [ sum, maximum ] = TNL::Algorithms::reduce< Device >( 0, n, fetch, MultipleReductions< Plus, Max >{} );
 * ```
 *
 * The tuple is created by \ref makeReductionTuple.
 */
template< typename... Values >
struct ReductionTuple;

template<>
struct ReductionTuple<>
{};

template< typename Value, typename... Values >
struct ReductionTuple< Value, Values... >
{
   Value head;
   ReductionTuple< Values... > tail;

   //! \brief Returns the component `I` of the tuple.
   template< std::size_t I >
   [[nodiscard]] __cuda_callable__
   constexpr auto&
   get()
   {
      if constexpr( I == 0 )
         return head;
      else
         return tail.template get< I - 1 >();
   }

   //! \brief Returns the component `I` of the tuple.
   template< std::size_t I >
   [[nodiscard]] __cuda_callable__
   constexpr const auto&
   get() const
   {
      if constexpr( I == 0 )
         return head;
      else
         return tail.template get< I - 1 >();
   }
};

//! \brief Creates an empty \ref ReductionTuple.
[[nodiscard]] __cuda_callable__
constexpr ReductionTuple<>
makeReductionTuple()
{
   return {};
}

//! \brief Creates a \ref ReductionTuple with the given components.
template< typename Value, typename... Values >
[[nodiscard]] __cuda_callable__
constexpr ReductionTuple< Value, Values... >
makeReductionTuple( const Value& value, const Values&... values )
{
   return { value, makeReductionTuple( values... ) };
}

/**
 * \brief A value with its argument (e.g. the index of the element) for the reductions with an argument
 * combined by \ref MultipleReductions, see \ref WithArgument.
 */
template< typename Value, typename Index >
struct ValueWithArgument
{
   Value value;
   Index argument;
};

/**
 * \brief Adapts a reduction with an argument (e.g. \ref TNL::MaxWithArg) to a reduction of
 * \ref ValueWithArgument, so that it can be combined with other reductions by \ref MultipleReductions.
 *
 * The identity has the identity of the value of `ReductionWithArgument` and the largest argument,
 * so it does not win the ties of the reductions preferring the smaller arguments.
 */
template< typename ReductionWithArgument >
struct WithArgument
{
   template< typename Value, typename Index >
   [[nodiscard]] __cuda_callable__
   constexpr ValueWithArgument< Value, Index >
   operator()( ValueWithArgument< Value, Index > a, const ValueWithArgument< Value, Index >& b ) const
   {
      ReductionWithArgument{}( a.value, b.value, a.argument, b.argument );
      return a;
   }

   template< typename Result >
   [[nodiscard]] __cuda_callable__
   static constexpr Result
   getIdentity()
   {
      return getIdentity( static_cast< Result* >( nullptr ) );
   }

private:
   template< typename Value, typename Index >
   [[nodiscard]] __cuda_callable__
   static constexpr ValueWithArgument< Value, Index >
   getIdentity( ValueWithArgument< Value, Index >* /*tag*/ )
   {
      return { ReductionWithArgument::template getIdentity< Value >(), std::numeric_limits< Index >::max() };
   }
};

/**
 * \brief Combines several reductions into one reduction of \ref ReductionTuple, so that all of them
 * are computed in one pass over the data.
 *
 * The component `I` of the tuples is reduced by the reduction function object `Reductions[ I ]`
 * (e.g. \ref TNL::Plus, \ref TNL::Max, or \ref WithArgument for the reductions with an argument),
 * the identity is the tuple of their identities. Since the product of monoids is a monoid, the
 * combined reduction can be used wherever a reduction function object is accepted, e.g. by
 * \ref TNL::Algorithms::reduce, \ref TNL::Algorithms::Segments::reduceAllSegments or
 * \ref TNL::Algorithms::Segments::multiReduceAllSegments. The fetch function returns the tuple of
 * the values of all reductions for one element, typically created by \ref makeReductionTuple.
 *
 * Several reductions of one vector:
 *
 * ```cpp
 * auto fetch = [ = ] __cuda_callable__( Index i )
 * {
 *    const Real x = v[ i ];
 *    return makeReductionTuple( x, x, x * x, ValueWithArgument< Real, Index >{ abs( x ), i } );
 * };
 * auto [ sum, maximum, squaredNorm, maxAbs ] =
 *    reduce< Device >( 0, n, fetch, MultipleReductions< Plus, Max, Plus, WithArgument< MaxWithArg > >{} );
 * ```
 *
 * Different reductions of several vectors of the same size:
 *
 * ```cpp
 * auto fetch = [ = ] __cuda_callable__( Index i )
 * {
 *    return makeReductionTuple( u[ i ] * v[ i ], abs( r[ i ] ), mask[ i ] != 0 );
 * };
 * auto [ dot, maxResidual, anyMasked ] = reduce< Device >( 0, n, fetch, MultipleReductions< Plus, Max, LogicalOr >{} );
 * ```
 *
 * The reduction function objects must be stateless, they are default-constructed.
 */
template< typename... Reductions >
struct MultipleReductions
{
   template< typename... Values >
   [[nodiscard]] __cuda_callable__
   constexpr ReductionTuple< Values... >
   operator()( const ReductionTuple< Values... >& a, const ReductionTuple< Values... >& b ) const
   {
      static_assert(
         sizeof...( Values ) == sizeof...( Reductions ), "the number of values does not match the number of reductions" );
      return apply( a, b, std::index_sequence_for< Values... >{} );
   }

   template< typename Result >
   [[nodiscard]] __cuda_callable__
   static constexpr Result
   getIdentity()
   {
      return getIdentity( static_cast< Result* >( nullptr ) );
   }

private:
   template< typename... Values, std::size_t... I >
   [[nodiscard]] __cuda_callable__
   static constexpr ReductionTuple< Values... >
   apply( const ReductionTuple< Values... >& a, const ReductionTuple< Values... >& b, std::index_sequence< I... > /*indexes*/ )
   {
      return makeReductionTuple( static_cast< Values >( Reductions{}( a.template get< I >(), b.template get< I >() ) )... );
   }

   template< typename... Values >
   [[nodiscard]] __cuda_callable__
   static constexpr ReductionTuple< Values... >
   getIdentity( ReductionTuple< Values... >* /*tag*/ )
   {
      static_assert(
         sizeof...( Values ) == sizeof...( Reductions ), "the number of values does not match the number of reductions" );
      return makeReductionTuple( Reductions::template getIdentity< Values >()... );
   }
};

}  // namespace TNL::Algorithms

// the structured bindings of ReductionTuple
namespace std {

template< typename... Values >
struct tuple_size< TNL::Algorithms::ReductionTuple< Values... > >
: public std::integral_constant< std::size_t, sizeof...( Values ) >
{};

template< std::size_t I, typename... Values >
struct tuple_element< I, TNL::Algorithms::ReductionTuple< Values... > >
{
   using type = std::tuple_element_t< I, std::tuple< Values... > >;
};

}  // namespace std
