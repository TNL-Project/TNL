// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page ConnectedComponentsOverview Overview of Connected Components Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all connected components functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * \section CCWhatIs What are Connected Components?
 *
 * A connected component is a maximal set of vertices such that each pair is
 * connected by a path. For directed graphs, the algorithm treats the input as
 * the underlying undirected graph — i.e. it computes **weakly** connected
 * components.
 *
 * On sequential and host backends the implementation uses a traversal-based
 * component expansion, while GPU backends use iterative label relaxation with
 * pointer jumping.
 *
 * \section CCVariants Function Variants
 *
 * All connected components functions follow this naming pattern:
 * `connectedComponents[If]`
 *
 * | Function                                       | Scope          | Edge filter | Overloads |
 * |------------------------------------------------|----------------|-------------|-----------|
 * | \ref connectedComponents (basic)               | Whole graph    | No          | 1         |
 * | \ref connectedComponents (edge predicate)      | Whole graph    | Yes         | 1         |
 * | \ref connectedComponents (vertex indexes)      | Vertex indexes | No          | 1         |
 * | \ref connectedComponents (idx + edge pred.)    | Vertex indexes | Yes         | 1         |
 * | \ref connectedComponentsIf                     | Vertex pred.   | No          | 1         |
 * | \ref connectedComponentsIf (edge predicate)    | Vertex pred.   | Yes         | 1         |
 *
 * \section CCSubgraphVariants Subgraph Variants
 *
 * Connected components can be computed on different subsets of the graph:
 *
 * Connected components can be computed on different subsets of the graph and
 * with optional edge filtering. These two dimensions combine independently:
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
 * \section CCLambdaSignatures Lambda Signatures
 *
 * \subsection CCEdgePredicate Edge predicate
 *
 * Decides if an edge can be traversed:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType source,
 *                                             typename Graph::IndexType target,
 *                                             typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * The algorithm treats the graph as undirected: for each stored edge (u,v)
 * it may also traverse in the reverse direction (v,u), using the same
 * predicate call with the natural orientation of the stored edge.
 *
 * \subsection CCVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section CCCommonParameters Common Parameters
 *
 * - **graph** — The input graph (const reference).
 * - **components** — Output vector for the representative vertex of each component.
 * - **launchConfig** — Configuration for parallel execution (optional).
 *
 * \section CCRelatedPages Related Pages
 */
// clang-format on

/**
 * \brief Finds connected components in a graph.
 *
 * The algorithm treats the input graph as the underlying undirected graph. In particular,
 * for directed graphs it computes weakly connected components.
 * On sequential and host backends it uses a traversal-based component expansion,
 * while GPU backends use iterative label relaxation with pointer jumping.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component representatives.
 * \param graph The graph on which the algorithm is performed.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc basic
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template< typename Graph, typename Vector >
void
connectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds connected components with edge filtering.
 *
 * The edge predicate decides if an edge can be traversed. It must provide
 * a call operator with the signature:
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType source, typename Graph::IndexType target,
 *                        typename Graph::ValueType weight ) -> bool
 * \endcode
 * The algorithm treats the graph as undirected: for each stored edge (u,v)
 * it may also traverse in the reverse direction (v,u), using the same
 * predicate call with the natural orientation of the stored edge.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store component representatives.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which the algorithm is performed.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc edge predicate
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< !IsArrayType< EdgePredicate >::value > >
void
connectedComponents(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds connected components in the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices outside of the induced subgraph are treated as absent and
 * receive component label \c -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store component representatives.
 * \param graph The graph on which the algorithm is performed.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc induced
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
connectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds connected components in the indexed-induced subgraph with edge filtering.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store component representatives.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which the algorithm is performed.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc induced edge predicate
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
connectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds connected components in the subgraph selected by a vertex predicate.
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
 * \tparam Vector The type of the vector used to store component representatives.
 * \param graph The graph on which the algorithm is performed.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc if
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
connectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Finds connected components in the predicate-induced subgraph with edge filtering.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge may be used.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store component representatives.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which the algorithm is performed.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param components The vector where the representative vertex of each component is stored.
 * \param launchConfig The configuration for graph traversal on parallel backends.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc if edge predicate
 *
 * See \ref ConnectedComponentsOverview for an overview of all connected components variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
connectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Graphs::Algorithms

#include "connectedComponents.hpp"
