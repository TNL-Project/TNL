// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page BFSOverview Overview of Breadth-first Search Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all breadth-first search (BFS) functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search) for more
 * details about the BFS algorithm.
 *
 * \section BFSWhatIs What is BFS?
 *
 * Breadth-first search traverses a graph layer by layer starting from a given
 * source vertex. The result is a distance vector where each entry holds the
 * number of edges on the shortest path from the source, or \c -1 for
 * unreachable vertices.
 *
 * \section BFSVariants Function Variants
 *
 * All BFS functions follow this naming pattern:
 * `breadthFirstSearch[If][WithVisitor]`
 * - The **If** suffix adds a vertex predicate parameter to select the active subgraph.
 * - The **WithVisitor** suffix adds a visitor callable invoked on each visited node.
 *
 * \subsection BFSBasicFunctions Basic BFS (no visitor)
 *
 * | Function                                    | Scope          | Edge filter |
 * |---------------------------------------------|----------------|-------------|
 * | \ref breadthFirstSearch (basic)             | Whole graph    | No          |
 * | \ref breadthFirstSearch (edge predicate)    | Whole graph    | Yes         |
 * | \ref breadthFirstSearch (vertex indexes)    | Vertex indexes | No          |
 * | \ref breadthFirstSearch (idx + edge pred.)  | Vertex indexes | Yes         |
 * | \ref breadthFirstSearchIf                   | Vertex pred.   | No          |
 * | \ref breadthFirstSearchIf (edge predicate)  | Vertex pred.   | Yes         |
 *
 * \subsection BFSVisitorFunctions BFS with visitor
 *
 * | Function                                          | Scope          | Edge filter |
 * |---------------------------------------------------|----------------|-------------|
 * | \ref breadthFirstSearchWithVisitor (basic)        | Whole graph    | No          |
 * | \ref breadthFirstSearchWithVisitor (edge pred.)   | Whole graph    | Yes         |
 * | \ref breadthFirstSearchWithVisitor (vertex idx.)  | Vertex indexes | No          |
 * | \ref breadthFirstSearchWithVisitor (idx + edge)   | Vertex indexes | Yes         |
 * | \ref breadthFirstSearchIfWithVisitor              | Vertex pred.   | No          |
 * | \ref breadthFirstSearchIfWithVisitor (edge pred.) | Vertex pred.   | Yes         |
 *
 * \section BFSSubgraphVariants Subgraph Variants
 *
 * BFS can operate on different subsets of the graph and with optional edge
 * filtering. These two dimensions combine independently:
 *
 * | Variant         | Vertices processed                           | Parameter added      |
 * |-----------------|----------------------------------------------|----------------------|
 * | **Whole graph** | All vertices                                 | None                 |
 * | **Indexed**     | Only vertices listed in a vertex-index array | `vertexIndexes`      |
 * | **If**          | Vertices selected by a vertex predicate      | `vertexPredicate`    |
 *
 * | Edge filter | Edges usable                          | Parameter added   |
 * |-------------|---------------------------------------|-------------------|
 * | **None**    | All edges are traversed               | None              |
 * | **Yes**     | Only edges allowed by the predicate   | `edgePredicate`   |
 *
 * Vertices outside the active subgraph keep distance \c -1 in the output.
 *
 * \section BFSLambdaSignatures Lambda Signatures
 *
 * \subsection BFSEdgePredicate Edge predicate
 *
 * Decides if a traversed edge may be used:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType source,
 *                                             typename Graph::IndexType target,
 *                                             typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * \subsection BFSVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \subsection BFSVisitor Visitor callable
 *
 * Invoked upon visiting each node:
 *
 * ```cpp
 * auto visitor = [=] __cuda_callable__( typename Graph::IndexType node,
 *                                       typename Graph::IndexType distance ) { ... };
 * ```
 *
 * \section BFSCommonParameters Common Parameters
 *
 * - **graph** — The input graph (const reference).
 * - **start** — The index of the source vertex.
 * - **distances** — Output vector for distances from the source vertex.
 * - **launchConfig** — Configuration for parallel execution (optional).
 */
// clang-format on

/**
 * \brief Performs breadth-first search (BFS) on the given graph starting from the specified node.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search) for more details about the BFS algorithm.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs basic
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 * [=] __cuda_callable__( typename Graph::IndexType source, typename Graph::IndexType target,
 *                        typename Graph::ValueType weight ) -> bool
 * \endcode
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs induced
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs induced edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * The start vertex must belong to the induced subgraph. Vertices not selected
 * by the predicate are never traversed and keep distance \c -1 in the output.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs if
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs if edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 * \brief Performs breadth-first search (BFS) on the given graph with a visitor callback.
 *
 * The visitor is invoked upon visiting each node. It must accept two parameters:
 * the node index and its distance from the start node.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
 */
template< typename Graph, typename Vector, typename Visitor, typename = std::enable_if_t< ! IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with edge filtering and a visitor callback.
 *
 * The edge predicate decides if a traversed edge can be used, and the visitor is
 * invoked upon visiting each node. The edge predicate must provide a call operator
 * with the signature:
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType source, typename Graph::IndexType target,
 *                        typename Graph::ValueType weight ) -> bool
 * \endcode
 * The visitor must accept two parameters: the node index and its distance from the
 * start node.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename Visitor,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   EdgePredicate&& edgePredicate,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) on the induced subgraph with a visitor callback.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor induced
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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
 * \brief Performs BFS on the induced subgraph with edge filtering and a visitor callback.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices and the
 * start vertex must belong to the induced subgraph. The edge predicate has the
 * same requirements as in the whole-graph overload, and the visitor is invoked
 * upon visiting each node.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor induced edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename EdgePredicate,
   typename Visitor,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs BFS on the predicate-induced subgraph with a visitor callback.
 *
 * The predicate must provide
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * and the start vertex must belong to the induced subgraph.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor if
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
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

/**
 * \brief Performs BFS on the predicate-induced subgraph with edge filtering and a visitor callback.
 *
 * The vertex predicate selects active vertices, the edge predicate decides if a
 * traversed edge may be used, and the visitor is invoked upon visiting each node.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store distances.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting node for BFS.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge can be traversed.
 * \param visitor The callable invoked upon visiting each node.
 * \param distances The vector where distances from the start node will be stored.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor if edge predicate
 *
 * See \ref BFSOverview for an overview of all breadth-first search variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate, typename Visitor >
void
breadthFirstSearchIfWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );
}  // namespace TNL::Graphs::Algorithms

#include "breadthFirstSearch.hpp"
