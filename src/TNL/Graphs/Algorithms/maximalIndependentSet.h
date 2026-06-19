// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page MaximalIndependentSetOverview Overview of Maximal Independent Set Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all maximal independent set functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * \section MISWhatIs What is a Maximal Independent Set?
 *
 * A maximal independent set (MIS) is a set of vertices such that no two
 * vertices in the set are adjacent (independence) and no vertex can be added
 * without violating independence (maximality). The implementation uses
 * deterministic Luby-style priority rounds. The output is a 0/1 mask where
 * value 1 marks vertices that belong to the MIS.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Maximal_independent_set) for more details.
 *
 * \section MISComputationFunctions Computation Functions
 *
 * | Function                                         | Scope          | Edge filter | Overloads |
 * |--------------------------------------------------|----------------|-------------|-----------|
 * | \ref maximalIndependentSet (basic)               | Whole graph    | No          | 1         |
 * | \ref maximalIndependentSet (edge predicate)      | Whole graph    | Yes         | 1         |
 * | \ref maximalIndependentSet (vertex indexes)      | Vertex indexes | No          | 1         |
 * | \ref maximalIndependentSet (idx + edge pred.)    | Vertex indexes | Yes         | 1         |
 * | \ref maximalIndependentSetIf                     | Vertex pred.   | No          | 1         |
 * | \ref maximalIndependentSetIf (edge predicate)    | Vertex pred.   | Yes         | 1         |
 *
 * \section MISVerificationFunctions Verification Functions
 *
 * | Function                                              | Scope          | Edge filter | Overloads |
 * |-------------------------------------------------------|----------------|-------------|-----------|
 * | \ref isMaximalIndependentSet (basic)                  | Whole graph    | No          | 1         |
 * | \ref isMaximalIndependentSet (edge predicate)         | Whole graph    | Yes         | 1         |
 * | \ref isMaximalIndependentSet (vertex indexes)         | Vertex indexes | No          | 1         |
 * | \ref isMaximalIndependentSet (idx + edge pred.)       | Vertex indexes | Yes         | 1         |
 * | \ref isMaximalIndependentSetIf                        | Vertex pred.   | No          | 1         |
 * | \ref isMaximalIndependentSetIf (edge predicate)       | Vertex pred.   | Yes         | 1         |
 *
 * \section MISSubgraphVariants Subgraph Variants
 *
 * MIS can be computed on different subsets of the graph and with optional
 * edge filtering. These two dimensions combine independently:
 *
 * | Variant         | Vertices processed                           | Parameter added       |
 * |-----------------|----------------------------------------------|-----------------------|
 * | **Whole graph** | All vertices                                 | None                  |
 * | **Indexed**     | Only vertices listed in a vertex-index array | `vertexIndexes`       |
 * | **If**          | Vertices selected by a vertex predicate      | `vertexPredicate`     |
 *
 * | Edge filter | Edges usable                          | Parameter added   |
 * |-------------|---------------------------------------|-------------------|
 * | **None**    | All edges are considered              | None              |
 * | **Yes**     | Only edges allowed by the predicate   | `edgePredicate`   |
 *
 * Vertices outside the active subgraph remain zero in the output mask.
 * Vertices connected only by blocked edges may coexist in the independent set.
 *
 * \section MISLambdaSignatures Lambda Signatures
 *
 * \subsection MISEdgePredicate Edge predicate
 *
 * Decides if an edge connects two vertices that are considered adjacent:
 *
 * ```cpp
 * auto edgePredicate = [=] __cuda_callable__( typename Graph::IndexType vertex,
 *                                             typename Graph::IndexType neighbor,
 *                                             typename Graph::ValueType weight ) -> bool { ... };
 * ```
 *
 * Vertices connected only by blocked edges may coexist in the independent set.
 *
 * \subsection MISVertexPredicate Vertex predicate
 *
 * Decides which vertices belong to the induced subgraph:
 *
 * ```cpp
 * auto vertexPredicate = [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool { ... };
 * ```
 *
 * \section MISCommonParameters Common Parameters
 *
 * - **graph** — The input undirected graph (const reference).
 * - **independentSet** — Output 0/1 mask (1 = vertex belongs to the MIS).
 */
// clang-format on

/**
 * \brief Finds a maximal independent set in an undirected graph.
 *
 * The implementation uses deterministic Luby-style priority rounds. The
 * output is a 0/1 mask where value 1 marks vertices that belong to the
 * maximal independent set.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \param graph The input undirected graph.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis basic
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename Vector >
void
maximalIndependentSet(
   const Graph& graph,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Finds a maximal independent set with edge filtering.
 *
 * The edge predicate decides if an edge connects two vertices that are
 * considered adjacent. It must provide a call operator with the signature:
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex, typename Graph::IndexType neighbor,
 *                        typename Graph::ValueType weight ) -> bool
 * \endcode
 * Vertices connected only by blocked edges may coexist in the independent set.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
void
maximalIndependentSet(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Finds a maximal independent set in the subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 * Vertices not listed in \e vertexIndexes are excluded from the subgraph and
 * remain zero in the output mask.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis induced
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
void
maximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Finds a maximal independent set in the indexed-induced subgraph with edge filtering.
 *
 * The edge predicate has the same requirements as in the whole-graph overload.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis induced edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
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
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Finds a maximal independent set in the subgraph defined by a vertex predicate.
 *
 * The predicate decides which vertices belong to the induced subgraph. It must
 * provide a call operator with the signature
 * \code
 * [=] __cuda_callable__( typename Graph::IndexType vertex ) -> bool
 * \endcode
 * The output is still a full-size 0/1 mask over the original graph.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis if
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
void
maximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Finds a maximal independent set in the predicate-induced subgraph with edge filtering.
 *
 * The vertex predicate selects active vertices and the edge predicate decides
 * if a traversed edge connects two adjacent vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis if edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
maximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the whole graph.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \param graph The input undirected graph.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis basic
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename Vector >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set
 * considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set with respect to the allowed edges.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template<
   typename Graph,
   typename Vector,
   typename EdgePredicate,
   typename = std::enable_if_t< ! IsArrayType< EdgePredicate >::value > >
bool
isMaximalIndependentSet(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * subgraph induced by the given vertex indexes.
 *
 * The entries in \e vertexIndexes must be unique valid graph vertices.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set in the induced subgraph.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis induced
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Vector,
   typename = std::enable_if_t< IsArrayType< VertexIndexes >::value > >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * indexed-induced subgraph considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexIndexes The type of the array containing the vertex indexes.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexIndexes The array of vertex indexes defining the induced subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set in the induced subgraph with respect to the allowed edges.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis induced edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
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
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * subgraph selected by a vertex predicate.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set in the predicate-induced subgraph.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis if
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector >
bool
isMaximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set in the
 * predicate-induced subgraph considering only allowed edges.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexPredicate The type of the vertex predicate callable.
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \tparam EdgePredicate The type of the edge predicate callable.
 * \param graph The input undirected graph.
 * \param vertexPredicate The callable deciding which vertices belong to the subgraph.
 * \param edgePredicate The callable deciding if an edge connects adjacent vertices.
 * \param independentSet The 0/1 mask to verify.
 * \return true If the mask defines a maximal independent set in the predicate-induced subgraph with respect to the allowed
 * edges.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis if edge predicate
 *
 * See \ref MaximalIndependentSetOverview for an overview of all maximal independent set variants.
 */
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isMaximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms

#include "maximalIndependentSet.hpp"
