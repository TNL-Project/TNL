// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page StronglyConnectedComponentsOverview Overview of Strongly Connected Components Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all strongly connected components
 * functions, helping to understand the differences between variants and choose
 * the right function for your needs.
 *
 * \section SCCWhatIs What are Strongly Connected Components?
 *
 * A strongly connected component is a maximal set of vertices in a directed
 * graph such that each pair is connected by a directed path in both
 * directions. This is meaningful only for directed graphs; for undirected
 * graphs use \ref connectedComponents instead.
 *
 * \section SCVariants Function Variants
 *
 * All strongly connected components functions follow this naming pattern:
 * `stronglyConnectedComponents[If]`
 *
 * | Function                                                | Scope          | Edge filter | Overloads |
 * |---------------------------------------------------------|----------------|-------------|-----------|
 * | \ref stronglyConnectedComponents (basic)                | Whole graph    | No          | 1         |
 * | \ref stronglyConnectedComponents (edge predicate)       | Whole graph    | Yes         | 1         |
 * | \ref stronglyConnectedComponents (vertex indexes)       | Vertex indexes | No          | 1         |
 * | \ref stronglyConnectedComponents (idx + edge pred.)     | Vertex indexes | Yes         | 1         |
 * | \ref stronglyConnectedComponentsIf                      | Vertex pred.   | No          | 1         |
 * | \ref stronglyConnectedComponentsIf (edge predicate)     | Vertex pred.   | Yes         | 1         |
 *
 * \section SCSubgraphVariants Subgraph Variants
 *
 * Strongly connected components can be computed on different subsets of the
 * graph and with optional edge filtering. These two dimensions combine
 * independently:
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
 * Vertices outside the active subgraph receive component label \c -1.
 *
 * \section SCLambdaSignatures Lambda Signatures
 *
 * \subsection SCEdgePredicate Edge predicate
 *
 * Decides if an edge can be traversed:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType source,
 *                  typename Graph::IndexType target,
 *                  typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * \subsection SCVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section SCCommonParameters Common Parameters
 *
 * - **graph** — The input directed graph (const reference).
 * - **components** — Output vector for the component label of each vertex.
 * - **launchConfig** — Configuration for parallel execution (optional).
 *
 * \section SCRelatedPages Related Pages
 */
// clang-format on

/**
 * \brief Finds strongly connected components in a directed graph.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component labels.
 * \param graph The directed graph on which the algorithm is performed.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc basic
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
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
 * [=] __cuda_callable__( typename Graph::IndexType source, typename Graph::IndexType target,
 *                  typename Graph::ValueType weight ) -> bool
 * \endcode
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The directed graph on which the algorithm is performed.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc edge predicate
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename Enable = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store component labels.
 * \param graph The directed graph on which the algorithm is performed.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc induced
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename Enable = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store component labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The directed graph on which the algorithm is performed.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc induced edge predicate
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename Enable = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
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
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * Vertices not selected by the predicate are treated as absent and
 * receive component label \c -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store component labels.
 * \param graph The directed graph on which the algorithm is performed.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc if
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store component labels.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The directed graph on which the algorithm is performed.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the labels of strongly connected components are stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc if edge predicate
 *
 * See \ref StronglyConnectedComponentsOverview for an overview of all strongly connected components variants.
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
