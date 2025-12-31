// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

//! \brief Structure for specifying graph type.
template< bool Directed >
struct GraphOrientation
{
   [[nodiscard]] static constexpr bool
   isDirected()
   {
      return Directed;
   }

   [[nodiscard]] static std::string
   getSerializationType()
   {
      if( isDirected() )
         return "Directed";
      return "Undirected";
   }
};

//! \brief Undirected graph type.
struct UndirectedGraph : GraphOrientation< false >
{};

//! \brief Directed graph type.
struct DirectedGraph : GraphOrientation< true >
{};

}  // namespace TNL::Graphs
