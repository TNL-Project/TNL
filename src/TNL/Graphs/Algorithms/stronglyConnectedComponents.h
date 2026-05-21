// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Finds strongly connected components in a directed graph.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component labels.
 * \param graph is the directed graph on which the algorithm is performed.
 * \param components is the vector where the labels of strongly connected components are stored.
 * \param launchConfig is the configuration for graph traversal on parallel backends.
 */
template< typename Graph, typename Vector >
void
stronglyConnectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "stronglyConnectedComponents.hpp"
