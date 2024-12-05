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

template< typename Lambda, typename Is, typename = void >
struct can_accept_impl : std::false_type
{};

template< typename Lambda, std::size_t... Is >
struct can_accept_impl< Lambda,
                        std::index_sequence< Is... >,
                        decltype( std::declval< Lambda >()( ( (void) Is, any_argument{} )... ), void() ) > : std::true_type
{};

template< typename Lambda, std::size_t N >
struct can_accept : can_accept_impl< Lambda, std::make_index_sequence< N > >
{};

template< typename Lambda, std::size_t Max, std::size_t N, typename = void >
struct lambda_details_impl : lambda_details_impl< Lambda, Max, N - 1 >
{};

template< typename Lambda, std::size_t Max, std::size_t N >
struct lambda_details_impl< Lambda, Max, N, std::enable_if_t< can_accept< Lambda, N >::value > >
{
   static constexpr bool
   isVariadic()
   {
      return ( N == Max );
   }

   static constexpr std::size_t
   argumentCount()
   {
      return N;
   }
};

template< typename Lambda, std::size_t Max = 50 >
struct LambdaDetails : lambda_details_impl< Lambda, Max, Max >
{};

}  //namespace TNL::detail
