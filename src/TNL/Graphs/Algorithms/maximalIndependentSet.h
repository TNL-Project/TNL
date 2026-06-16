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
 * \brief Finds a maximal independent set with edge filtering.
 *
 * The edge predicate decides if an edge connects two vertices that are
 * considered adjacent. It must provide a call operator with the signature:
 * \code
 * bool operator()( typename Graph::IndexType vertex, typename Graph::IndexType neighbor,
 *                  typename Graph::ValueType weight ) const;
 * \endcode
 * Vertices connected only by blocked edges may coexist in the independent set.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< !IsArrayType< EdgePredicate >::value > >
void
maximalIndependentSet( const Graph& graph, EdgePredicate&& edgePredicate, Vector& independentSet );

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
 * \brief Finds a maximal independent set in the indexed-induced subgraph with edge filtering.
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
maximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& independentSet );

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
 * \brief Finds a maximal independent set in the predicate-induced subgraph with edge filtering.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge connects two adjacent vertices.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
maximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the whole graph.
 */
template< typename Graph, typename Vector >
bool
isMaximalIndependentSet( const Graph& graph, const Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set
 * considering only allowed edges.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< !IsArrayType< EdgePredicate >::value > >
bool
isMaximalIndependentSet( const Graph& graph, EdgePredicate&& edgePredicate, const Vector& independentSet );

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
 * indexed-induced subgraph considering only allowed edges.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * subgraph selected by a vertex predicate.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isMaximalIndependentSetIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& independentSet );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * predicate-induced subgraph considering only allowed edges.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isMaximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet );

}  // namespace TNL::Graphs::Algorithms

#include "maximalIndependentSet.hpp"
