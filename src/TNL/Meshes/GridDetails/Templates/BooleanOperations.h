// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace TNL::Meshes::Templates {

/**
 * One of the possible implementation of the conjunction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/conjunction
 */

template< class... >
struct conjunction : std::true_type
{};

template< class Type >
struct conjunction< Type > : Type
{};

template< class Head, class... Tail >
struct conjunction< Head, Tail... > : std::conditional_t< bool( Head::value ), conjunction< Tail... >, Head >
{};

template< class... Types >
constexpr bool conjunction_v = conjunction< Types... >::value;

/**
 * One of the possible implementation of the conjunction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/disjunction
 */

template< class... >
struct disjunction : std::false_type
{};

template< class Type >
struct disjunction< Type > : Type
{};

template< class Head, class... Tail >
struct disjunction< Head, Tail... > : std::conditional_t< bool( Head::value ), Head, disjunction< Tail... > >
{};

template< class... Types >
constexpr bool disjunction_v = disjunction< Types... >::value;

}  // namespace TNL::Meshes::Templates
