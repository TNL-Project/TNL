// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

/**
 * \brief Namespace for TNL allocators.
 *
 * All TNL allocators must satisfy the requirements imposed by the
 * [Allocator concept](https://en.cppreference.com/w/cpp/named_req/Allocator)
 * from STL.
 */
namespace TNL::Allocators {

/**
 * \brief Allocator for the host memory space -- alias for \ref std::allocator.
 */
template< class T >
using Host = std::allocator< T >;

}  // namespace TNL::Allocators
