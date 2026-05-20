// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Finds connected components in a graph.
 *
 * The algorithm treats the input graph as the underlying undirected graph. In particular,
 * for directed graphs it computes weakly connected components.
 * On sequential and host backends it uses a traversal-based component expansion,
 * while GPU backends use iterative label relaxation with pointer jumping.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component representatives.
 * \param graph is the graph on which the algorithm is performed.
 * \param components is the vector where the representative vertex of each component is stored.
 * \param launchConfig is the configuration for graph traversal on parallel backends.
 */
template< typename Graph, typename Vector >
void
connectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "connectedComponents.hpp"
