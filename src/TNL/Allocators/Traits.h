// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace TNL {

/**
 * \brief A trait-class that determines if an allocator allocates data that are
 * directly accessible from the host code without the need for explicit copy
 * operations.
 *
 * The trait is equivalent to \ref std::true_type by default and each
 * definition of an allocator that does not meet the above condition must be
 * accompanied by a template specialization for this trait.
 *
 * \tparam Allocator a type to checky
 */
template< typename Allocator >
struct allocates_host_accessible_data : public std::true_type
{};

/**
 * \brief A helper variable template for \ref allocates_host_accessible_data.
 */
template< typename Allocator >
constexpr bool allocates_host_accessible_data_v = allocates_host_accessible_data< Allocator >::value;

}  // namespace TNL
