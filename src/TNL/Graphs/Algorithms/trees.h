// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page TreeDetectionOverview Overview of Tree and Forest Detection Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all tree and forest detection functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * \section TDWhatIs What are Trees and Forests?
 *
 * A **tree** is a connected acyclic graph. A **forest** is a disjoint union
 * of trees. The functions below verify whether a given graph (or subgraph)
 * has these properties.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more
 * details about trees and forests in graph theory.
 *
 * \section TDVariants Function Variants
 *
 * All tree and forest detection functions follow these naming patterns:
 * - `isTree[If]` — check if the (sub)graph is a single tree
 * - `isForest[If]` — check if the (sub)graph is a forest with auto-detected roots
 * - `isForestWithRoots[If]` — check if the (sub)graph is a forest with explicit root candidates
 *
 * \subsection TDIsTreeFunctions isTree
 *
 * | Function                                    | Scope          | Edge filter | Overloads |
 * |---------------------------------------------|----------------|-------------|-----------|
 * | \ref isTree (basic)                         | Whole graph    | No          | 1         |
 * | \ref isTree (edge predicate)                | Whole graph    | Yes         | 1         |
 * | \ref isTree (vertex indexes)                | Vertex indexes | No          | 1         |
 * | \ref isTree (idx + edge pred.)              | Vertex indexes | Yes         | 1         |
 * | \ref isTreeIf                               | Vertex pred.   | No          | 1         |
 * | \ref isTreeIf (edge predicate)              | Vertex pred.   | Yes         | 1         |
 *
 * \subsection TDIsForestFunctions isForest
 *
 * | Function                                    | Scope          | Edge filter | Overloads |
 * |---------------------------------------------|----------------|-------------|-----------|
 * | \ref isForest (basic)                       | Whole graph    | No          | 1         |
 * | \ref isForest (edge predicate)              | Whole graph    | Yes         | 1         |
 * | \ref isForest (vertex indexes)              | Vertex indexes | No          | 1         |
 * | \ref isForest (idx + edge pred.)            | Vertex indexes | Yes         | 1         |
 * | \ref isForestIf                             | Vertex pred.   | No          | 1         |
 * | \ref isForestIf (edge predicate)            | Vertex pred.   | Yes         | 1         |
 *
 * \subsection TDIsForestWithRootsFunctions isForestWithRoots
 *
 * | Function                                          | Scope          | Edge filter | Overloads |
 * |---------------------------------------------------|----------------|-------------|-----------|
 * | \ref isForestWithRoots (basic)                    | Whole graph    | No          | 1         |
 * | \ref isForestWithRoots (edge predicate)           | Whole graph    | Yes         | 1         |
 * | \ref isForestWithRoots (vertex indexes)           | Vertex indexes | No          | 1         |
 * | \ref isForestWithRoots (idx + edge pred.)         | Vertex indexes | Yes         | 1         |
 * | \ref isForestWithRootsIf                          | Vertex pred.   | No          | 1         |
 * | \ref isForestWithRootsIf (edge predicate)         | Vertex pred.   | Yes         | 1         |
 *
 * \section TDSubgraphVariants Subgraph Variants
 *
 * Tree and forest detection can operate on different subsets of the graph
 * and with optional edge filtering. These two dimensions combine independently:
 *
 * | Variant         | Vertices processed                           | Parameter added       |
 * |-----------------|----------------------------------------------|-----------------------|
 * | **Whole graph** | All vertices                                 | None                  |
 * | **Indexed**     | Only vertices listed in a vertex-index array | `vertexIndexes`       |
 * | **If**          | Vertices selected by a vertex predicate      | `vertexPredicate`     |
 *
 * | Edge filter | Edges usable                          | Parameter added   |
 * |-------------|---------------------------------------|-------------------|
 * | **None**    | All edges are traversed               | None              |
 * | **Yes**     | Only edges allowed by the predicate   | `edgePredicate`   |
 *
 * For **isTree**, the edge count check uses only active edges (between active
 * vertices and allowed by the edge predicate) instead of the total graph edge
 * count. The expected count is \c n_active - 1, where \c n_active is the number
 * of active vertices.
 *
 * \section TDDetails Variant Details
 *
 * - **isTree** checks that the (sub)graph is connected and has exactly
 *   \c n_active - 1 active edges, starting the traversal from the given
 *   \e start vertex.
 * - **isForest (auto roots)** detects roots automatically by finding unvisited
 *   active vertices during BFS traversal.
 * - **isForestWithRoots** uses the provided root candidates as traversal
 *   starting points for each component.
 *
 * \section TDLambdaSignatures Lambda Signatures
 *
 * \subsection TDEdgePredicate Edge predicate
 *
 * Decides if an edge can be traversed:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType source,
 *                                              typename Graph::IndexType target,
 *                                              typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * \subsection TDVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section TDCommonParameters Common Parameters
 *
 * - **graph** — The input graph (const reference).
 * - **start** — The starting vertex for tree check.
 * - **roots** — Vector of root candidates for forest check.
 * - **vertexIndexes** — Array of vertex indexes defining the induced subgraph.
 * - **vertexPredicate** — Callable deciding which vertices belong to the subgraph.
 * - **edgePredicate** — Callable deciding if an edge can be traversed.
 */
// clang-format on

/**
 * \brief Checks if the given graph is a tree.
 *
 * The graph is a tree if it is connected and has exactly n-1 edges,
 * starting the traversal from the given \e start vertex.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \return true If the graph is a tree.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start = 0 );

/**
 * \brief Checks if the given graph is a tree considering only allowed edges.
 *
 * The edge predicate decides if an edge can be traversed. It must provide
 * a call operator with the signature:
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType source, typename Graph::IndexType target,
 *                        typename Graph::ValueType weight ) -> bool
 * \endcode
 *
 * The edge count check uses only active edges instead of the total graph
 * edge count.
 *
 * \tparam Graph The type of the graph.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the graph is a tree with respect to the allowed edges.
 * \return false Otherwise.
 */
template< typename Graph, typename EdgePredicate, typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
bool
isTree( const Graph& graph, typename Graph::IndexType start, EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the subgraph induced by the given vertex indexes is a tree.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph. Vertices outside of the
 * induced subgraph are treated as absent.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \return true If the induced subgraph is a tree.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexIndexes, typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isTree( const Graph& graph, typename Graph::IndexType start, const VertexIndexes& vertexIndexes );

/**
 * \brief Checks if the indexed-induced subgraph is a tree considering only allowed edges.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the induced subgraph is a tree with respect to the allowed edges.
 * \return false Otherwise.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isTree(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the subgraph selected by a vertex predicate is a tree.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * The start vertex must belong to the induced subgraph.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \return true If the predicate-induced subgraph is a tree.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate >
bool
isTreeIf( const Graph& graph, typename Graph::IndexType start, VertexPredicate&& vertexPredicate );

/**
 * \brief Checks if the predicate-induced subgraph is a tree considering only allowed edges.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge may be used.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the predicate-induced subgraph is a tree with respect to the allowed edges.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isTreeIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the given graph is a forest with auto-detected roots.
 *
 * Roots of each tree component are detected automatically by finding unvisited
 * vertices during BFS traversal.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about forests.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isForest( const Graph& graph );

/**
 * \brief Checks if the given graph is a forest considering only allowed edges.
 *
 * Roots are detected automatically. The edge predicate decides if an edge can
 * be traversed.
 *
 * \tparam Graph The type of the graph.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the graph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template< typename Graph, typename EdgePredicate, typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
bool
isForest( const Graph& graph, EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the subgraph induced by the given vertex indexes is a forest.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Roots are detected automatically among the active vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \param graph The graph to check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \return true If the induced subgraph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexIndexes, typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isForest( const Graph& graph, const VertexIndexes& vertexIndexes );

/**
 * \brief Checks if the indexed-induced subgraph is a forest considering only allowed edges.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 * Roots are detected automatically among the active vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the induced subgraph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isForest( const Graph& graph, const VertexIndexes& vertexIndexes, EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the subgraph selected by a vertex predicate is a forest.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * Roots are detected automatically among the active vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \param graph The graph to check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \return true If the predicate-induced subgraph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate >
bool
isForestIf( const Graph& graph, VertexPredicate&& vertexPredicate );

/**
 * \brief Checks if the predicate-induced subgraph is a forest considering only allowed edges.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge may be used. Roots are detected automatically.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph to check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \return true If the predicate-induced subgraph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isForestIf( const Graph& graph, VertexPredicate&& vertexPredicate, EdgePredicate&& edgePredicate );

/**
 * \brief Checks if the given graph is a forest using the provided root candidates.
 *
 * Each root candidate starts a BFS traversal for one tree component.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about forests.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename Vector >
bool
isForestWithRoots( const Graph& graph, const Vector& roots );

/**
 * \brief Checks if the given graph is a forest with edge filtering using the provided root candidates.
 *
 * The edge predicate decides if an edge can be traversed.
 *
 * \tparam Graph The type of the graph.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the graph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template<
   typename Graph,
   typename EdgePredicate,
   typename Vector,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
bool
isForestWithRoots( const Graph& graph, EdgePredicate&& edgePredicate, const Vector& roots );

/**
 * \brief Checks if the subgraph induced by the given vertex indexes is a forest using the provided root candidates.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the induced subgraph is a forest.
 * \return false Otherwise.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isForestWithRoots( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& roots );

/**
 * \brief Checks if the indexed-induced subgraph is a forest with edge filtering using the provided root candidates.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the induced subgraph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename EdgePredicate,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isForestWithRoots( const Graph& graph, const VertexIndexes& vertexIndexes, EdgePredicate&& edgePredicate, const Vector& roots );

/**
 * \brief Checks if the subgraph selected by a vertex predicate is a forest using the provided root candidates.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the predicate-induced subgraph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isForestWithRootsIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& roots );

/**
 * \brief Checks if the predicate-induced subgraph is a forest with edge filtering using the provided root candidates.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge may be used.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the predicate-induced subgraph is a forest with respect to the allowed edges.
 * \return false Otherwise.
 */
template< typename Graph, typename VertexPredicate, typename EdgePredicate, typename Vector >
bool
isForestWithRootsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& roots );

}  // namespace TNL::Graphs::Algorithms

#include "trees.hpp"
