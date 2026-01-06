// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Performs breadth-first search (BFS) on the given graph starting from the specified node.
 *
 * See. [Wikipedia page](https://en.wikipedia.org/wiki/Breadth-first_search) for more details about the BFS algorithm.
 *
 * \tparam Graph Type of the graph.
 * \tparam Vector Type of the vector used to store distances.
 * \param graph is the graph on which BFS is performed.
 * \param start is the starting node for BFS.
 * \param distances is the vector where distances from the start node will be stored.
 * \param launchConfig is the configuration for launching the segments traversal.
 */
template< typename Graph, typename Vector >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the given graph starting from the specified node.
 *
 * See. [Wikipedia page](https://en.wikipedia.org/wiki/Breadth-first_search) for more details about the BFS algorithm.
 *
 * \tparam Graph Type of the graph.
 * \tparam Vector Type of the vector used to store distances.
 * \param graph is the graph on which BFS is performed.
 * \param start is the starting node for BFS.
 * \param distances is the vector where distances from the start node will be stored.
 * \param visitor is a callable object that will be invoked upon visiting each node. It should accept two parameters:
 *        the node index and its distance from the start node.
 * \param launchConfig is the configuration for launching the segments traversal.
 */
template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   Visitor&& visitor,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Graphs::Algorithms

#include "breadthFirstSearch.hpp"
