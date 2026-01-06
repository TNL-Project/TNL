// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Computes single source shortest paths using parallel algorithm.
 * See [Wikipedia page](https://en.wikipedia.org/wiki/Shortest_path_problem) for more details about the
 * algorithm.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph is the graph on which the algorithm is performed.
 * \param start is the starting node for the algorithm.
 * \param distances is the vector where distances from the start node will be stored.
 * \param launchConfig is the configuration for launching the segments traversal.
 */
template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Graphs::Algorithms

#include "singleSourceShortestPath.hpp"
