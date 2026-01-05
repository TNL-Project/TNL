// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
struct Graph;

//! \brief This checks if given type is matrix.
[[nodiscard]] constexpr std::false_type
isGraph( ... )
{
   return {};
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
[[nodiscard]] constexpr std::true_type
isGraph( const Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >& )
{
   return {};
}

}  // namespace TNL::Graphs
