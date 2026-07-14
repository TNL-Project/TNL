// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Finds strongly connected components in a directed graph.
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
