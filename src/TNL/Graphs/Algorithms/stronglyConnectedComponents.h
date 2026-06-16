// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

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

/**
 * \brief Finds strongly connected components with edge filtering.
 *
 * The edge predicate decides if an edge can be traversed. It must provide
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
stronglyConnectedComponents(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds strongly connected components in the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices outside of the induced subgraph are treated as absent and
 * receive component label \c -1 in the output.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
stronglyConnectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds strongly connected components in the indexed-induced subgraph with edge filtering.
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
stronglyConnectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds strongly connected components in the subgraph selected by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * Vertices not selected by the predicate are treated as absent and
 * receive component label \c -1 in the output.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
stronglyConnectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds strongly connected components in the predicate-induced subgraph with edge filtering.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge may be used.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
stronglyConnectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "stronglyConnectedComponents.hpp"
