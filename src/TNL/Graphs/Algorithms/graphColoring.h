// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page GraphColoringOverview Overview of Graph Coloring Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all graph coloring functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * \section GCWhatIs What is Graph Coloring?
 *
 * Graph coloring assigns integer labels (colors) to vertices so that no two
 * adjacent vertices share the same color. The goal is typically to use as few
 * colors as possible. Colors are zero-based in the output; inactive vertices
 * are marked by \c -1.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Graph_coloring) for more details.
 *
 * \section GCAlgorithmVariants Algorithm Variants
 *
 * Two coloring strategies are available:
 *
 * | Strategy                     | Function prefix       | Description                                         |
 * |------------------------------|-----------------------|-----------------------------------------------------|
 * | **Greedy** (speculative)     | `graphColoring`       | Each uncolored vertex proposes the smallest safe color; conflicts resolved by vertex priority |
 * | **Luby MIS**                 | `graphColoringLuby`   | Each color class is one maximal independent set on the remaining subgraph |
 *
 * \subsection GCGreedyFunctions Greedy Coloring
 *
 * | Function                                    | Scope          | Edge filter | Overloads |
 * |---------------------------------------------|----------------|-------------|-----------|
 * | \ref graphColoring (basic)                  | Whole graph    | No          | 1         |
 * | \ref graphColoring (edge predicate)         | Whole graph    | Yes         | 1         |
 * | \ref graphColoring (vertex indexes)         | Vertex indexes | No          | 1         |
 * | \ref graphColoring (idx + edge pred.)       | Vertex indexes | Yes         | 1         |
 * | \ref graphColoringIf                        | Vertex pred.   | No          | 1         |
 * | \ref graphColoringIf (edge predicate)       | Vertex pred.   | Yes         | 1         |
 *
 * \subsection GCLubyFunctions Luby-style MIS Coloring
 *
 * | Function                                    | Scope          | Edge filter | Overloads |
 * |---------------------------------------------|----------------|-------------|-----------|
 * | \ref graphColoringLuby (basic)              | Whole graph    | No          | 1         |
 * | \ref graphColoringLuby (edge predicate)     | Whole graph    | Yes         | 1         |
 * | \ref graphColoringLuby (vertex indexes)     | Vertex indexes | No          | 1         |
 * | \ref graphColoringLuby (idx + edge pred.)   | Vertex indexes | Yes         | 1         |
 * | \ref graphColoringLubyIf                    | Vertex pred.   | No          | 1         |
 * | \ref graphColoringLubyIf (edge predicate)   | Vertex pred.   | Yes         | 1         |
 *
 * \section GCVerification Verification Functions
 *
 * | Function                                    | Scope          | Edge filter | Overloads |
 * |---------------------------------------------|----------------|-------------|-----------|
 * | \ref isProperlyColored (basic)              | Whole graph    | No          | 1         |
 * | \ref isProperlyColored (edge predicate)     | Whole graph    | Yes         | 1         |
 * | \ref isProperlyColored (vertex indexes)     | Vertex indexes | No          | 1         |
 * | \ref isProperlyColored (idx + edge pred.)   | Vertex indexes | Yes         | 1         |
 * | \ref isProperlyColoredIf                    | Vertex pred.   | No          | 1         |
 * | \ref isProperlyColoredIf (edge predicate)   | Vertex pred.   | Yes         | 1         |
 *
 * \section GCSubgraphVariants Subgraph Variants
 *
 * Graph coloring can operate on different subsets of the graph and with
 * optional edge filtering. These two dimensions combine independently:
 *
 * | Variant         | Vertices processed                           | Parameter added       | Inactive marker |
 * |-----------------|----------------------------------------------|-----------------------|-----------------|
 * | **Whole graph** | All vertices                                 | None                  | N/A             |
 * | **Indexed**     | Only vertices listed in a vertex-index array | `vertexIndexes`       | \c -1           |
 * | **If**          | Vertices selected by a vertex predicate      | `vertexPredicate`     | \c -1           |
 *
 * | Edge filter | Edges usable                          | Parameter added   |
 * |-------------|---------------------------------------|-------------------|
 * | **None**    | All edges are considered              | None              |
 * | **Yes**     | Only edges allowed by the predicate   | `edgePredicate`   |
 *
 * Vertices connected only by blocked edges may receive the same color.
 *
 * \section GCLambdaSignatures Lambda Signatures
 *
 * \subsection GCEdgePredicate Edge predicate
 *
 * Decides if an edge connects two vertices that are considered adjacent for
 * coloring purposes:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType vertex,
 *                  typename Graph::IndexType neighbor, typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * Vertices connected only by blocked edges may receive the same color.
 *
 * \subsection GCVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section GCCommonParameters Common Parameters
 *
 * - **graph** — The input undirected graph (const reference).
 * - **colors** — Output vector for zero-based color labels (inactive vertices get \c -1).
 */
// clang-format on

/**
 * \brief Colors an undirected graph with zero-based integer labels by greedy algorithm.
 *
 * The implementation uses speculative rounds: every uncolored vertex proposes
 * the smallest color not used by already colored neighbors, and conflicts
 * among equal-color neighbors are resolved deterministically by vertex
 * priority.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename Vector >
void
graphColoring(
   const Graph& graph,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors an undirected graph with edge filtering by greedy algorithm.
 *
 * The edge predicate decides if an edge connects two vertices that are
 * considered adjacent for coloring purposes. It must provide a call operator
 * with the signature:
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex, typename Graph::IndexType neighbor,
 *                  typename Graph::ValueType weight ) -> bool
 * \endcode
 * Vertices connected only by blocked edges may receive the same color.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
void
graphColoring(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the subgraph induced by the given vertex indexes with zero-based labels by greedy algorithm.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are marked by -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoring(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the indexed-induced subgraph with edge filtering by greedy algorithm.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoring(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the subgraph defined by a vertex predicate with zero-based labels by greedy algorithm.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * Vertices not selected by the predicate are marked by -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the predicate-induced subgraph with edge filtering by greedy algorithm.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge connects two adjacent vertices for coloring purposes.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors an undirected graph by repeated Luby-style MIS extraction.
 *
 * Each color class is built by finding one maximal independent set on the
 * still-uncolored subgraph and assigning one color to all of its vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename Vector >
void
graphColoringLuby(
   const Graph& graph,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors an undirected graph with edge filtering by repeated Luby-style MIS extraction.
 *
 * The edge predicate has the same requirements as in the greedy overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
void
graphColoringLuby(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the subgraph induced by the given vertex indexes by repeated Luby-style MIS extraction.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are marked by -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoringLuby(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the indexed-induced subgraph with edge filtering by Luby-style MIS extraction.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
graphColoringLuby(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the subgraph defined by a vertex predicate by repeated Luby-style MIS extraction.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * Vertices not selected by the predicate are marked by -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringLubyIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors the predicate-induced subgraph with edge filtering by Luby-style MIS extraction.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge connects two adjacent vertices for coloring purposes.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The output vector of zero-based color labels.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringLubyIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that all color labels are non-negative and adjacent vertices differ.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector holding color labels.
 * \param graph The input undirected graph.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper (no adjacent vertices share a color and all labels are non-negative).
 */
template< typename Graph, typename Vector >
bool
isProperlyColored(
   const Graph& graph,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the coloring is proper considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector holding color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper with respect to the allowed edges.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
bool
isProperlyColored(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the active part of the given coloring is proper on the
 * subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes must be marked by -1.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector holding color labels.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper on the induced subgraph.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isProperlyColored(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the coloring is proper on the indexed-induced subgraph
 * considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector holding color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper on the induced subgraph with respect to the allowed edges.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isProperlyColored(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the active part of the given coloring is proper on the
 * subgraph selected by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * Vertices not selected by the predicate must be marked by -1.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector holding color labels.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper on the predicate-induced subgraph.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isProperlyColoredIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the coloring is proper on the predicate-induced subgraph
 * considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector holding color labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param colors The vector of color labels to verify.
 * \return true If the coloring is proper on the predicate-induced subgraph with respect to the allowed edges.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isProperlyColoredIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms

#include "graphColoring.hpp"
