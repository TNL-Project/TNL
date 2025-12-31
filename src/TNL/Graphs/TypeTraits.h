// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
struct Graph;

//! \brief This checks if given type is matrix.
[[nodiscard]] constexpr std::false_type
isGraph( ... )
{
   return {};
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] constexpr std::true_type
isGraph( const Graph< Value, Device, Index, Orientation, AdjacencyMatrix >& )
{
   return {};
}

}  // namespace TNL::Graphs
