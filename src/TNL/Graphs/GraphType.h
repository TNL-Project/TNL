// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

//! \brief Structure for specifying graph type.
template< bool Directed >
struct GraphType
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
struct UndirectedGraph : GraphType< false >
{};

//! \brief Directed graph type.
struct DirectedGraph : GraphType< true >
{};

}  // namespace TNL::Graphs
