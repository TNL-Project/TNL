// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace TNL {
//! \brief Namespace for TNL execution models
namespace Devices {

struct Sequential
{
   //! Not used by any sequential algorithm, only for compatibility with parallel execution models.
   struct LaunchConfiguration
   {};
};

/***
 * \brief Returns true if the device type is Sequential.
 */
template< typename Device >
constexpr bool isSequential() { return std::is_same< Device, Sequential >::value; }

}  // namespace Devices
}  // namespace TNL
