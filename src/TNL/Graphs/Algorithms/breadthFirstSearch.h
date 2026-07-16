// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page BFSOverview Overview of Breadth-first Search Functions
 *
 * Breadth-first search traverses a graph layer by layer from a source vertex,
 * producing a distance vector (edge count to source, or \c -1 if unreachable).
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search) for more
 * details about the BFS algorithm.
 *
 * | Function                                          | Visitor | Predecessors | Description                          |
 * |---------------------------------------------------|---------|--------------|--------------------------------------|
 * | \ref breadthFirstSearch                           | No      | No           | Plain BFS, returns distances         |
 * | \ref breadthFirstSearchWithVisitor                | Yes     | No           | BFS with a visitor callback per node |
 * | \ref breadthFirstSearchWithPredecessors           | No      | Yes          | BFS with predecessor tracking       |
 * | \ref breadthFirstSearchWithVisitorAndPredecessors | Yes     | Yes          | BFS with visitor + predecessors |
 *
 // AGENT-TODO: Add description of the visitor callable.
 * To run BFS on a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * breadthFirstSearch( sg, start, distances );
 * ```
 */
// clang-format on

/**
 * \brief Performs breadth-first search (BFS) on the given graph starting from the specified node.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search) for more details about the BFS algorithm.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument. Vertices outside the active subgraph
 * keep distance \c -1 in the output.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs basic
 */
template< typename Graph, typename Vector >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with a visitor callback.
 *
 * The visitor is invoked upon visiting each node. It must accept two parameters:
 * the node index and its distance from the start node.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor
 */
template<
   typename Graph,
   typename Vector,
   typename Visitor,
   typename Enable = std::enable_if_t< ! IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with predecessor tracking.
 *
 * In addition to distances, this function computes a predecessor vector where
 * \c predecessors[v] is the parent of vertex \c v in the BFS tree.  The start
 * vertex and any unreachable vertices have predecessor \c -1.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam PredecessorVector The type of the vector used to store predecessors.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param distances The vector where distances from the start node will be stored.
 * \param predecessors The vector where predecessor indices will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 * \param deterministic If \c true, the predecessor of each vertex is chosen as
 *   the smallest source index among all valid parents in the same BFS layer.
 *   If \c false (default), the predecessor is whichever thread wins the
 *   atomic update — faster but non-reproducible between runs.
 */
template< typename Graph, typename Vector, typename PredecessorVector >
void
breadthFirstSearchWithPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   PredecessorVector& predecessors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration(),
   bool deterministic = false );

/**
 * \brief Performs breadth-first search (BFS) with a visitor callback and
 *   predecessor tracking.
 *
 * Combines the functionality of \ref breadthFirstSearchWithVisitor and
 * \ref breadthFirstSearchWithPredecessors.  The visitor is invoked exactly
 * once per discovered vertex (excluding the start vertex).
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam PredecessorVector The type of the vector used to store predecessors.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param predecessors The vector where predecessor indices will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 * \param deterministic If \c true, predecessors are chosen deterministically
 *   (smallest source index per layer).  See
 *   \ref breadthFirstSearchWithPredecessors.
 */
template<
   typename Graph,
   typename Vector,
   typename PredecessorVector,
   typename Visitor,
   typename Enable = std::enable_if_t< ! IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitorAndPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   PredecessorVector& predecessors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration(),
   bool deterministic = false );
}  // namespace TNL::Graphs::Algorithms

#include "breadthFirstSearch.hpp"
