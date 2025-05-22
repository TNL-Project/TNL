// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL::Algorithms {

template< typename Device >
struct Reduction3D;

template<>
struct Reduction3D< Devices::Sequential >
{
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static constexpr void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int m, int n, Output result );
};

template<>
struct Reduction3D< Devices::Host >
{
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int m, int n, Output result );
};

template<>
struct Reduction3D< Devices::Cuda >
{
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int m, int n, Output hostResult );
};

}  // namespace TNL::Algorithms

#include "Reduction3D.hpp"
