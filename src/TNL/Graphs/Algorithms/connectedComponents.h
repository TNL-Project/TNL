// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Finds connected components in a graph.
 *
 * The algorithm treats the input graph as the underlying undirected graph. In
 * particular, for directed graphs it computes weakly connected components.
 * On sequential and host backends it uses a traversal-based component expansion,
 * while GPU backends use iterative label relaxation with pointer jumping.
 *
 * To operate on a subgraph (vertex predicate, edge predicate, or both),
 * construct a \ref SubGraph via \ref makeSubGraph and pass it as the \e graph
 * argument.  Vertices outside the active subgraph receive component label
 * \c -1 in the output.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector used to store component representatives.
 * \param graph The graph on which the algorithm is performed.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc basic
 */
template< typename Graph, typename Vector >
void
connectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "connectedComponents.hpp"
