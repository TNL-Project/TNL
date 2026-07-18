// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page StronglyConnectedComponentsOverview Overview of Strongly Connected Components Functions
 *
 * Strongly connected components (SCCs) partition a directed graph into maximal
 * subgraphs in which every vertex is reachable from every other vertex.  TNL
 * uses a pivot-based algorithm that performs forward and backward BFS from a
 * pivot vertex.  Component labels start at \c 1; inactive vertices receive
 * \c -1.  A directed (non-symmetric) graph is required (\c static_assert
 * enforced).
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Strongly_connected_component)
 * for more details about strongly connected components.
 *
 * | Function                              | Description                                          |
 * |---------------------------------------|------------------------------------------------------|
 * | \ref stronglyConnectedComponents      | Labels each vertex with its SCC (labels start at 1)  |
 *
 * To run SCC on a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * stronglyConnectedComponents( sg, components );
 * ```
 */
// clang-format on

/**
 * \brief Finds strongly connected components in a directed graph.
 *
 * See \ref StronglyConnectedComponentsOverview for an overview and algorithm
 * details.
 *
 * To operate on a subgraph (vertex predicate, edge predicate, or both),
 * construct a \ref SubGraph via \ref makeSubGraph and pass it as the \e graph
 * argument.  Vertices outside the active subgraph receive component label
 * \c -1 in the output.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector used to store component labels.
 * \param graph The directed graph on which the algorithm is performed.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc basic
 */
template< typename Graph, typename Vector >
void
stronglyConnectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "stronglyConnectedComponents.hpp"
