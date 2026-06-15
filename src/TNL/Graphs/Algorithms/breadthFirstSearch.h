// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

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
 * \brief Performs breadth-first search (BFS) with edge filtering.
 *
 * The edge predicate decides if a traversed edge can be used. It must provide
 * a call operator with the signature:
 * \code
 * bool operator()( typename Graph::IndexType source, typename Graph::IndexType target,
 *                  typename Graph::ValueType weight ) const;
 * \endcode
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< !IsArrayType< EdgePredicate >::value > >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph. Vertices outside of the
 * induced subgraph are treated as absent, so they are never traversed and keep
 * distance \c -1 in the output.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs BFS on the induced subgraph with edge filtering.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the subgraph selected by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * The start vertex must belong to the induced subgraph. Vertices not selected
 * by the predicate are never traversed and keep distance \c -1 in the output.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
breadthFirstSearchIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs BFS on the predicate-induced subgraph with edge filtering.
 *
 * The vertex predicate selects active vertices and edge predicate decides if a
 * traversed edge may be used.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
breadthFirstSearchIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
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
template<
   typename Graph,
   typename Vector,
   typename Visitor,
   typename = std::enable_if_t< !IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the induced subgraph given by vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename Visitor,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the induced subgraph selected by a vertex predicate.
 *
 * The predicate must provide
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * and the start vertex must belong to the induced subgraph.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename Visitor >
void
breadthFirstSearchIfWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "breadthFirstSearch.hpp"
