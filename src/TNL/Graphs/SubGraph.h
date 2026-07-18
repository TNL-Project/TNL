// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/TypeTraits.h>

#include "Graph.h"
#include "GraphView.h"
#include "TypeTraits.h"

namespace TNL::Graphs {

/**
 * \brief SubGraph combines a graph with vertex and edge filters.
 *
 * SubGraph is a lightweight, non-owning view-like object that wraps a const
 * reference to a graph together with two predicates:
 * - a **vertex filter** that decides which vertices are active, and
 * - an **edge filter** that decides which edges may be traversed.
 *
 * Both filters default to "accept all" when omitted. SubGraph is cheap to
 * copy (it stores only a matrix view and the filter callables by value) and
 * is usable in \c __cuda_callable__ contexts when the filters are
 * device-callable.
 *
 * SubGraph implements the same read-only interface as \ref GraphBase
 * (\c getVertexCount, \c getAdjacencyMatrixView, \c getVertex, etc.) and,
 * crucially, \c getView / \c getConstView return a SubGraph copy. This means
 * the existing traversal functions (\ref forAllEdges, \ref forEdges, ...)
 * dispatch to the \ref TNL::Graphs::detail::TraversingOperations
 * "TraversingOperations<SubGraph>" specialization which applies the filters
 * transparently — no algorithm-level branching is needed.
 *
 * In addition, SubGraph provides:
 * - \c vertexExists(vertex) — tests the vertex filter, and
 * - \c edgeExists(source, target, weight) — tests the edge filter.
 *
 * Use \ref makeSubGraph factory functions to create SubGraph instances:
 * ```cpp
 * auto sg1 = makeSubGraph( graph );                                    // full graph
 * auto sg2 = makeSubGraph( graph, vertexPredicate );                   // vertex filter only
 * auto sg3 = makeSubGraph( graph, {}, edgePredicate );                 // edge filter only
 * auto sg4 = makeSubGraph( graph, vertexPredicate, edgePredicate );    // both filters
 * auto sg5 = makeSubGraph( graph, vertexIndexes );                     // vertex mask from indexes
 * auto sg6 = makeSubGraph( graph, vertexIndexes, edgePredicate );      // mask + edge filter
 * ```
 *
 * \par Example
 * \snippet Graphs/SubGraphExample_VertexFilter.cpp vertex filter
 *
 * \par Example
 * \snippet Graphs/SubGraphExample_EdgeFilter.cpp edge filter
 *
 * \par Example
 * \snippet Graphs/SubGraphExample_Traverse.cpp traverse subgraph
 *
 * \tparam Graph_ The underlying graph type (Graph, GraphView, or another SubGraph).
 * \tparam VertexFilter Callable `(IndexType) -> bool` or mask array view.
 * \tparam EdgeFilter Callable `(IndexType, IndexType, ValueType) -> bool`.
 */
template< typename Graph_, typename VertexFilter, typename EdgeFilter >
class SubGraph
{
public:
   //! \brief Type of the underlying graph.
   using GraphType = std::decay_t< Graph_ >;

   //! \brief Type for indexing vertices.
   using IndexType = typename GraphType::IndexType;

   //! \brief Type for edge weights.
   using ValueType = typename GraphType::ValueType;

   //! \brief Device where the graph operates.
   using DeviceType = typename GraphType::DeviceType;

   //! \brief Type of the adjacency matrix view.
   using AdjacencyMatrixView = typename GraphType::AdjacencyMatrixView;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = typename GraphType::ConstAdjacencyMatrixView;

   //! \brief Type of the adjacency matrix (view, since SubGraph does not own the matrix).
   using AdjacencyMatrixType = ConstAdjacencyMatrixView;

   //! \brief Type of constant view of the underlying graph.
   using ConstGraphView = typename GraphType::ConstViewType;

   //! \brief Type of the graph orientation.
   using GraphOrientation = typename GraphType::GraphOrientation;

   //! \brief Type of the graph nodes view.
   using VertexView = typename GraphType::VertexView;

   //! \brief Type of constant graph nodes view.
   using ConstVertexView = typename GraphType::ConstVertexView;

   /**
    * \brief SubGraph IS the view type.
    *
    * SubGraph is lightweight (matrix view + two filter callables) and
    * copyable, so \c getView / \c getConstView simply return a copy of
    * \c *this.  This lets \ref traverse.hpp dispatch to the
    * \c TraversingOperations<SubGraph> specialization.
    */
   using ViewType = SubGraph< Graph_, VertexFilter, EdgeFilter >;

   //! \brief ConstViewType is the same as ViewType (SubGraph is already read-only).
   using ConstViewType = SubGraph< Graph_, VertexFilter, EdgeFilter >;

   static constexpr bool
   isDirected();

   static constexpr bool
   isUndirected();

   /**
    * \brief Constructs a SubGraph from a graph and two filters.
    *
    * \param graph The underlying graph.
    * \param vertexFilter Predicate `(IndexType) -> bool` selecting active vertices.
    * \param edgeFilter Predicate `(IndexType, IndexType, ValueType) -> bool` selecting traversable edges.
    */
   SubGraph( const GraphType& graph, VertexFilter vertexFilter, EdgeFilter edgeFilter );

   /**
    * \brief Constructs a SubGraph from a graph view and two filters.
    *
    * This constructor is used internally by \ref MaskedSubGraph to construct
    * a SubGraph without access to the original Graph object.
    */
   SubGraph( ConstGraphView graphView, VertexFilter vertexFilter, EdgeFilter edgeFilter );

   //! \brief Copy constructor (needed for getView / getConstView).
   SubGraph( const SubGraph& ) = default;

   //! \brief Move constructor.
   SubGraph( SubGraph&& ) = default;

   //! \brief Copy-assignment is deleted (consistent with GraphBase).
   SubGraph&
   operator=( const SubGraph& ) = delete;

   //! \brief Move-assignment is deleted (consistent with GraphBase).
   SubGraph&
   operator=( SubGraph&& ) = delete;

   //! \brief Returns the number of vertices in the underlying graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexCount() const;

   //! \brief Returns the number of edges in the underlying graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getEdgeCount() const;

   //! \brief Returns the const adjacency matrix view of the underlying graph.
   [[nodiscard]] __cuda_callable__
   const ConstAdjacencyMatrixView&
   getAdjacencyMatrixView() const;

   //! \brief Returns the adjacency matrix (view) of the underlying graph.
   [[nodiscard]] __cuda_callable__
   const AdjacencyMatrixType&
   getAdjacencyMatrix() const;

   /**
    * \brief Returns a modifiable view of the graph (a copy of \c *this).
    *
    * Since SubGraph is already a lightweight view, \c getView returns a copy.
    * The copy carries the same filters, so traversal functions dispatch to
    * \c TraversingOperations<SubGraph> which applies them transparently.
    */
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   /**
    * \brief Returns a constant view of the graph (a copy of \c *this).
    *
    * \see getView
    */
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   //! \brief Returns the constant view of the graph node with given index.
   [[nodiscard]] __cuda_callable__
   ConstVertexView
   getVertex( IndexType vertexIdx ) const;

   //! \brief Returns the degree of the given vertex.
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexDegree( IndexType vertexIdx ) const;

   /**
    * \brief Tests whether a vertex exists in the induced subgraph.
    * \param vertex The vertex index to test.
    * \return \c true if the vertex passes the vertex filter.
    */
   [[nodiscard]] __cuda_callable__
   bool
   vertexExists( IndexType vertex ) const;

   /**
    * \brief Tests whether an edge exists in the induced subgraph.
    * \param source The source vertex index.
    * \param target The target vertex index.
    * \param weight The edge weight.
    * \return \c true if both endpoints are active and the edge passes the edge filter.
    */
   [[nodiscard]] __cuda_callable__
   bool
   edgeExists( IndexType source, IndexType target, const ValueType& weight ) const;

   //! \brief Returns the vertex filter.
   [[nodiscard]] __cuda_callable__
   const VertexFilter&
   getVertexFilter() const;

   //! \brief Returns the edge filter.
   [[nodiscard]] __cuda_callable__
   const EdgeFilter&
   getEdgeFilter() const;

   /**
    * \brief Materializes this SubGraph into a standalone \ref Graph.
    *
    * The returned graph owns its adjacency matrix and contains only the
    * vertices and edges that pass the filters of this SubGraph. The vertex
    * count is preserved (inaktivní vrcholy zůstávají v matici, ale bez
    * odchozích hran); vertex indices are identical to the underlying graph.
    *
    * \par Semantics
    * - **Inactive vertices** (filtered out by the vertex filter) are kept
    *   as empty rows in the result — their identity is preserved.
    * - **Filtered edges** are omitted entirely.
    * - **Edges touching inactive vertices** (either as source or as target)
    *   are omitted — the vertex filter is applied to both endpoints of every
    *   edge during traversal, so the materialized graph is an induced
    *   subgraph on the active vertex set.
    *
    * \return A new \ref Graph owning the materialized adjacency matrix.
    */
   [[nodiscard]] Graph< ValueType, DeviceType, IndexType, GraphOrientation >
   materialize() const;

protected:
   ConstGraphView graphView_;
   VertexFilter vertexFilter_;
   EdgeFilter edgeFilter_;
};

/// Tag type for selecting the edge-filter-only overload of \ref makeSubGraph.
struct EdgeOnlyTag
{};

/// Tag instance for selecting the edge-filter-only overload of \ref makeSubGraph.
inline constexpr EdgeOnlyTag edgeOnly{};

// ---------------------------------------------------------------------------
// Factory function declarations (non-indexed)
// ---------------------------------------------------------------------------

/**
 * \brief Creates a SubGraph from a graph with no filtering (full graph).
 *
 * \param graph The input graph.
 * \return A SubGraph accepting all vertices and all edges.
 */
template< typename Graph >
auto
makeSubGraph( const Graph& graph );

/**
 * \brief Creates a SubGraph from a graph with a vertex filter only.
 *
 * \param graph The input graph.
 * \param vertexFilter Callable `(IndexType) -> bool` selecting active vertices.
 * \return A SubGraph with the given vertex filter and all edges accepted.
 */
template<
   typename Graph,
   typename VertexFilter,
   typename Enable = std::enable_if_t< ! IsArrayType< std::decay_t< VertexFilter > >::value > >
auto
makeSubGraph( const Graph& graph, VertexFilter&& vertexFilter );

/**
 * \brief Creates a SubGraph from a graph with an edge filter only (tag dispatch).
 *
 * Use \ref edgeOnly to disambiguate: \c makeSubGraph(graph, edgeOnly, edgeFilter).
 */
template< typename Graph, typename EdgeFilter >
auto
makeSubGraph( const Graph& graph, EdgeOnlyTag, EdgeFilter&& edgeFilter );

/**
 * \brief Creates a SubGraph from a graph with both vertex and edge filters.
 *
 * \param graph The input graph.
 * \param vertexFilter Callable `(IndexType) -> bool` selecting active vertices.
 * \param edgeFilter Callable `(IndexType, IndexType, ValueType) -> bool` selecting edges.
 * \return A SubGraph with the given filters.
 */
template<
   typename Graph,
   typename VertexFilter,
   typename EdgeFilter,
   typename Enable = std::enable_if_t< ! IsArrayType< std::decay_t< VertexFilter > >::value > >
auto
makeSubGraph( const Graph& graph, VertexFilter&& vertexFilter, EdgeFilter&& edgeFilter );

}  // namespace TNL::Graphs

#include "SubGraph.hpp"
#include "MaskedSubGraph.h"
