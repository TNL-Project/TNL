// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

/**
 * \brief Template structure for specifying graph orientation (directed or undirected).
 *
 * This template serves as the base for defining graph types. It provides compile-time
 * information about whether a graph is directed or undirected through the template parameter.
 *
 * \tparam Directed Boolean indicating if the graph is directed (true) or undirected (false).
 *
 * \see DirectedGraph, UndirectedGraph
 */
template< bool Directed >
struct GraphOrientation
{
   /**
    * \brief Checks if the graph is directed.
    * \return True if the graph is directed, false if undirected.
    */
   [[nodiscard]] static constexpr bool
   isDirected()
   {
      return Directed;
   }

   /**
    * \brief Returns a string representation of the graph type for serialization.
    * \return "Directed" for directed graphs, "Undirected" for undirected graphs.
    */
   [[nodiscard]] static std::string
   getSerializationType()
   {
      if( isDirected() )
         return "Directed";
      return "Undirected";
   }
};

/**
 * \brief Type tag for undirected graphs.
 *
 * An undirected graph is a graph where edges have no direction. If there is an edge
 * between vertices u and v, it can be traversed in both directions.
 *
 * \see DirectedGraph, GraphOrientation
 */
struct UndirectedGraph : GraphOrientation< false >
{};

/**
 * \brief Type tag for directed graphs.
 *
 * A directed graph (digraph) is a graph where edges have a direction. An edge from
 * vertex u to vertex v can only be traversed from u to v, not from v to u (unless
 * there is a separate edge in the opposite direction).
 *
 * \see UndirectedGraph, GraphOrientation
 */
struct DirectedGraph : GraphOrientation< true >
{};

}  // namespace TNL::Graphs
