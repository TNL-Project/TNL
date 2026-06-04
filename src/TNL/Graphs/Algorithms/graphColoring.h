// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

/**
 * \brief Colors an undirected graph with zero-based integer labels by greedy algorithm.
 *
 * The implementation uses speculative rounds: every uncolored vertex proposes
 * the smallest color not used by already colored neighbors, and conflicts
 * among equal-color neighbors are resolved deterministically by vertex
 * priority.
 */
template< typename Graph, typename Vector >
void
graphColoring( const Graph& graph, Vector& colors );

/**
 * \brief Colors the subgraph induced by the given vertex indexes with zero-based labels by greedy algorithm.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are marked by -1 in the output.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoring( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& colors );

/**
 * \brief Colors the subgraph defined by a vertex predicate with zero-based labels by greedy algorithm.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * Vertices not selected by the predicate are marked by -1 in the output.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& colors );

/**
 * \brief Colors an undirected graph by repeated Luby-style MIS extraction.
 *
 * Each color class is built by finding one maximal independent set on the
 * still-uncolored subgraph and assigning one color to all of its vertices.
 */
template< typename Graph, typename Vector >
void
graphColoringLubi( const Graph& graph, Vector& colors );

/**
 * \brief Colors the subgraph induced by the given vertex indexes by repeated Luby-style MIS extraction.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are marked by -1 in the output.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoringLubi( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& colors );

/**
 * \brief Colors the subgraph defined by a vertex predicate by repeated Luby-style MIS extraction.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * Vertices not selected by the predicate are marked by -1 in the output.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringLubiIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& colors );

/**
 * \brief Checks that all color labels are non-negative and adjacent vertices differ.
 */
template< typename Graph, typename Vector >
bool
isProperlyColored( const Graph& graph, const Vector& colors );

/**
 * \brief Checks that the active part of the given coloring is proper on the
 * subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes must be marked by -1.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isProperlyColored( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& colors );

/**
 * \brief Checks that the active part of the given coloring is proper on the
 * subgraph selected by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * bool operator()( typename Graph::IndexType vertex ) const;
 * \endcode
 * Vertices not selected by the predicate must be marked by -1.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isProperlyColoredIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& colors );

}  // namespace TNL::Graphs::Algorithms

#include "graphColoring.hpp"
