// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page SSSPOverview Overview of Single-source Shortest Path Functions
 *
 * Single-source shortest path computes the minimum-cost distance from a source
 * vertex to every other vertex in a weighted graph, producing a distance vector
 * (\c -1 for unreachable vertices).
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Shortest_path_problem) for more
 * details about the shortest path problem.
 *
 * | Function                                          | Edge weight transform | Description                |
 * |---------------------------------------------------|------------------------|----------------------------|
 * | \ref singleSourceShortestPath (basic)             | No                     | Uses original edge weights |
 * | \ref singleSourceShortestPath (edge weight call.) | Yes                    | Transforms weights via call.|
 *
 * To run SSSP on a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * singleSourceShortestPath( sg, start, distances );
 * ```
 */
// clang-format on

/**
 * \brief Computes single source shortest paths using parallel algorithm.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Shortest_path_problem) for more details about the
 * algorithm.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp basic
 */
template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes single-source shortest paths with edge-weight transformation.
 *
 * The edge-weight callable must provide the signature:
 * \code
 * [=] __cuda_callable__( Index source, Index target, typename Graph::ValueType weight ) -> typename Graph::ValueType
 * \endcode
 * Returning infinity (for example `std::numeric_limits< ValueType >::infinity()`)
 * marks the edge as non-traversable.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgeWeightCallable The type of the edge-weight transformation callable.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param edgeWeightCallable The callable transforming edge weights during traversal.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp edge weight callable
 */
template<
   typename Graph,
   typename Vector,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType,
   typename Enable = std::enable_if_t< ! IsArrayType< EdgeWeightCallable >::value > >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "singleSourceShortestPath.hpp"
