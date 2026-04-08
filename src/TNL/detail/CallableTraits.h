// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <utility>
#include <type_traits>

namespace TNL::detail {

/***
 * Adopted from https://stackoverflow.com/questions/54389831/count-the-number-of-arguments-in-a-lambda
 */

struct any_argument
{
   template< typename T >
   operator T&&() const;
};

template< typename Callable, typename Is, typename = void >
struct can_accept_impl : std::false_type
{};

template< typename Callable, std::size_t... Is >
struct can_accept_impl< Callable,
                        std::index_sequence< Is... >,
                        decltype( std::declval< Callable >()( ( (void) Is, any_argument{} )... ), void() ) > : std::true_type
{};

template< typename Callable, std::size_t N >
struct can_accept : can_accept_impl< Callable, std::make_index_sequence< N > >
{};

/// @cond DOXY_IGNORE
template< typename Callable, std::size_t Max, std::size_t N, typename = void >
struct callable_details_impl : callable_details_impl< Callable, Max, N - 1 >
{};
/// @endcond

template< typename Callable, std::size_t Max, std::size_t N >
struct callable_details_impl< Callable, Max, N, std::enable_if_t< can_accept< Callable, N >::value > >
{
   static constexpr bool
   isVariadicCallable()
   {
      return N == Max;
   }

   static constexpr std::size_t
   callableArgumentCount()
   {
      return N;
   }
};

template< typename Callable, std::size_t Max = 50 >
struct CallableTraits : callable_details_impl< Callable, Max, Max >
{};

}  // namespace TNL::detail
