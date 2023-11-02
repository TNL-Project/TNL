// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>  // reduction functions like std::plus, std::logical_and, std::logical_or etc.

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL::Algorithms {

template< typename Device >
struct Reduction3D;

template<>
struct Reduction3D< Devices::Sequential >
{
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static constexpr void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int m, int n, Result* result );
};

template<>
struct Reduction3D< Devices::Host >
{
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int m, int n, Result* result );
};

template<>
struct Reduction3D< Devices::Cuda >
{
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int m, int n, Result* hostResult );
};

}  // namespace TNL::Algorithms

#include "Reduction3D.hpp"
