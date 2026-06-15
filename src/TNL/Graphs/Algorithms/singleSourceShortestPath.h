// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

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

/**
 * \brief Computes single-source shortest paths with edge-weight transformation.
 *
 * The edge-weight callable must provide the signature:
 * \code
 * typename Graph::ValueType operator()( Index source, Index target, typename Graph::ValueType weight ) const;
 * \endcode
 * Returning infinity (for example `std::numeric_limits< ValueType >::infinity()`)
 * marks the edge as non-traversable.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType,
   typename = std::enable_if_t< !IsArrayType< EdgeWeightCallable >::value > >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes single-source shortest paths on the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph. Vertices outside of the
 * induced subgraph are treated as absent and keep distance \c -1 in the output.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename Index = typename Graph::IndexType,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes single-source shortest paths on an induced subgraph with edge-weight transformation.
 *
 * The edge-weight callable has the same requirements as in the whole-graph overload.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes single-source shortest paths on the subgraph selected by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * The start vertex must belong to the induced subgraph and unselected vertices
 * keep distance \c -1 in the output.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename Index = typename Graph::IndexType >
void
singleSourceShortestPathIf(
   const Graph& graph,
   Index start,
   VertexPredicate&& vertexPredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes single-source shortest paths on a predicate-induced subgraph with edge-weight transformation.
 */
template<
   typename Graph,
   typename VertexPredicate,
   typename Vector,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType >
void
singleSourceShortestPathIf(
   const Graph& graph,
   Index start,
   VertexPredicate&& vertexPredicate,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "singleSourceShortestPath.hpp"
