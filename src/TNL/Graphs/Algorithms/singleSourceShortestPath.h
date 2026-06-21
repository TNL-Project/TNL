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
 * \tableofcontents
 *
 * This page provides an overview of all single-source shortest path (SSSP)
 * functions, helping to understand the differences between variants and choose
 * the right function for your needs.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Shortest_path_problem) for more
 * details about shortest path algorithms.
 *
 * \section SSSPWhatIs What is SSSP?
 *
 * Single-source shortest path computes the minimum-cost distance from a given
 * source vertex to every other vertex in a weighted graph. The result is a
 * distance vector where each entry holds the shortest-path distance, or
 * \c -1 for unreachable vertices.
 *
 * \section SSSPVariants Function Variants
 *
 * All SSSP functions follow this naming pattern:
 * `singleSourceShortestPath[If]`
 *
 * | Function                                           | Scope          | Edge weight transform |
 * |----------------------------------------------------|----------------|-----------------------|
 * | \ref singleSourceShortestPath (basic)              | Whole graph    | No                    |
 * | \ref singleSourceShortestPath (edge weight call.)  | Whole graph    | Yes                   |
 * | \ref singleSourceShortestPath (vertex indexes)     | Vertex indexes | No                    |
 * | \ref singleSourceShortestPath (idx + edge weight)  | Vertex indexes | Yes                   |
 * | \ref singleSourceShortestPathIf                    | Vertex pred.   | No                    |
 * | \ref singleSourceShortestPathIf (edge weight call.)| Vertex pred.   | Yes                   |
 *
 * \section SSSPSubgraphVariants Subgraph Variants
 *
 * SSSP can operate on different subsets of the graph and with optional
 * edge-weight transformation. These two dimensions combine independently:
 *
 * | Variant         | Vertices processed                           | Parameter added       |
 * |-----------------|----------------------------------------------|-----------------------|
 * | **Whole graph** | All vertices                                 | None                  |
 * | **Indexed**     | Only vertices listed in a vertex-index array | `vertexIndexes`       |
 * | **If**          | Vertices selected by a vertex predicate      | `vertexPredicate`     |
 *
 * | Edge weight transform | Effect                                  | Parameter added         |
 * |-----------------------|-----------------------------------------|-------------------------|
 * | **None**              | Original edge weights are used          | None                    |
 * | **Yes**               | Weights are transformed by a callable   | `edgeWeightCallable`    |
 *
 * Vertices outside the active subgraph keep distance \c -1 in the output.
 *
 * \section SSSPLambdaSignatures Lambda Signatures
 *
 * \subsection SSSPEdgeWeightCallable Edge weight callable
 *
 * Transforms or filters edge weights during traversal. Returning infinity
 * (e.g. `std::numeric_limits< ValueType >::infinity()`) marks the edge as
 * non-traversable:
 *
 * ```cpp
 * auto edgeWeightCallable = [=] __cuda_callable__( typename Graph::IndexType source,
 *                                                  typename Graph::IndexType target,
 *                                                  typename Graph::ValueType weight ) -> typename Graph::ValueType { ... };
 * ```
 *
 * \subsection SSSPVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section SSSPCommonParameters Common Parameters
 *
 * - **graph** — The input graph (const reference).
 * - **start** — The index of the source vertex.
 * - **distances** — Output vector for shortest-path distances from the source.
 * - **launchConfig** — Configuration for parallel execution (optional).
 */
// clang-format on

/**
 * \brief Computes single source shortest paths using parallel algorithm.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Shortest_path_problem) for more details about the
 * algorithm.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp basic
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
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
 * \tparam Graph The type of the graph.
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
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
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

/**
 * \brief Computes single-source shortest paths on the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph. Vertices outside of the
 * induced subgraph are treated as absent and keep distance \c -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp induced
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename Index = typename Graph::IndexType,
   typename Enable = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgeWeightCallable The type of the edge-weight transformation callable.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgeWeightCallable The callable transforming edge weights during traversal.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp induced edge weight callable
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType,
   typename Enable = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
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
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * The start vertex must belong to the induced subgraph and unselected vertices
 * keep distance \c -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp if
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgeWeightCallable The type of the edge-weight transformation callable.
 * \tparam Index The type used for indexing elements in the graph.
 * \param graph The graph on which the algorithm is performed.
 * \param start The starting node for the algorithm.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgeWeightCallable The callable transforming edge weights during traversal.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp if edge weight callable
 *
 * See \ref SSSPOverview for an overview of all single-source shortest path variants.
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
