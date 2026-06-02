// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Finds a maximal independent set in an undirected graph.
 *
 * The implementation uses deterministic Luby-style priority rounds. The
 * output is a 0/1 mask where value 1 marks vertices that belong to the
 * maximal independent set.
 */
template< typename Graph, typename Vector >
void
maximalIndependentSet( const Graph& graph, Vector& independentSet );

/**
 * \brief Finds a maximal independent set in the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are excluded from the subgraph and
 * remain zero in the output mask.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
maximalIndependentSet( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& independentSet );

/**
 * \brief Finds a maximal independent set in the subgraph defined by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * The output is still a full-size 0/1 mask over the original graph.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
maximalIndependentSetIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the whole graph.
 */
template< typename Graph, typename Vector >
bool
isMaximalIndependentSet( const Graph& graph, const Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isMaximalIndependentSet( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * subgraph selected by a vertex predicate.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isMaximalIndependentSetIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& independentSet );

}  // namespace TNL::Graphs::Algorithms

#include "maximalIndependentSet.hpp"
