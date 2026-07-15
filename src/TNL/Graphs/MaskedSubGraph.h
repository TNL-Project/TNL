// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>

#include "SubGraph.h"

namespace TNL::Graphs {

/**
 * \brief MaskedSubGraph is a SubGraph that owns its vertex mask.
 *
 * While \ref SubGraph is a non-owning view that stores filter callables by
 * value, MaskedSubGraph owns the boolean mask array that defines which
 * vertices are active. It is created by the indexed overloads of
 * \ref makeSubGraph:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexIndexes );              // mask from indexes
 * auto sg = makeSubGraph( graph, vertexIndexes, edgePredicate ); // mask + edge filter
 * ```
 *
 * MaskedSubGraph provides the same read-only interface as SubGraph
 * (\c getVertexCount, \c getAdjacencyMatrixView, \c isActive, etc.).
 * Crucially, \c getView / \c getConstView return a lightweight \ref SubGraph
 * that references the internal mask. This means the existing traversal
 * functions (\ref forAllEdges, \ref forEdges, ...) and algorithms (BFS,
 * connected components, etc.) dispatch to the
 * \ref TNL::Graphs::detail::TraversingOperations
 * "TraversingOperations<SubGraph>" specialization and apply the mask
 * transparently.
 *
 * \tparam Graph_ The underlying graph type (Graph or GraphView).
 * \tparam EdgeFilter Callable `(IndexType, IndexType, ValueType) -> bool`.
 *
 * \par Example
 * \snippet Graphs/SubGraphExample_MaskedSubGraph.cpp masked subgraph
 */
template< typename Graph_, typename EdgeFilter >
class MaskedSubGraph
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

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = typename GraphType::ConstAdjacencyMatrixView;

   //! \brief Type of the adjacency matrix (view, since MaskedSubGraph does not own the matrix).
   using AdjacencyMatrixType = ConstAdjacencyMatrixView;

   //! \brief Type of constant view of the underlying graph.
   using ConstGraphView = typename GraphType::ConstViewType;

   //! \brief Type of the graph orientation.
   using GraphOrientation = typename GraphType::GraphOrientation;

   //! \brief Type of constant graph nodes view.
   using ConstVertexView = typename GraphType::ConstVertexView;

   //! \brief Type of the owned mask array.
   using MaskArray = Containers::Array< IndexType, DeviceType, IndexType >;

   //! \brief Const view type of the mask array.
   using ConstMaskView = typename MaskArray::ConstViewType;

   //! \brief Vertex filter type used in the SubGraph view.
   using VertexFilter = detail::MaskVertexFilter< ConstMaskView >;

   /**
    * \brief The view type returned by \c getView / \c getConstView.
    *
    * This is a \ref SubGraph referencing the internal mask, so traversal
    * functions dispatch to \c TraversingOperations<SubGraph>.
    */
   using ViewType = SubGraph< GraphType, VertexFilter, EdgeFilter >;

   //! \brief ConstViewType is the same as ViewType (SubGraph is already read-only).
   using ConstViewType = SubGraph< GraphType, VertexFilter, EdgeFilter >;

   static constexpr bool
   isDirected();

   static constexpr bool
   isUndirected();

   /**
    * \brief Constructs an MaskedSubGraph from a graph, vertex indexes, and an edge filter.
    *
    * \param graph The underlying graph.
    * \param vertexIndexes Array of vertex indexes that should be active.
    * \param edgeFilter Predicate `(IndexType, IndexType, ValueType) -> bool`.
    */
   template< typename VertexIndexes >
   MaskedSubGraph( const GraphType& graph, const VertexIndexes& vertexIndexes, EdgeFilter edgeFilter );

   //! \brief Copy constructor.
   MaskedSubGraph( const MaskedSubGraph& ) = default;

   //! \brief Move constructor.
   MaskedSubGraph( MaskedSubGraph&& ) = default;

   //! \brief Copy-assignment is deleted.
   MaskedSubGraph&
   operator=( const MaskedSubGraph& ) = delete;

   //! \brief Move-assignment is deleted.
   MaskedSubGraph&
   operator=( MaskedSubGraph&& ) = delete;

   //! \brief Returns the number of vertices in the underlying graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexCount() const;

   //! \brief Returns the number of edges in the underlying graph.
   [[nodiscard]] IndexType
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
    * \brief Returns a modifiable view (a SubGraph referencing the internal mask).
    *
    * The returned SubGraph holds a MaskVertexFilter that references the mask
    * owned by this MaskedSubGraph. The view is valid as long as this
    * MaskedSubGraph is alive.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a constant view (a SubGraph referencing the internal mask).
    *
    * \see getView
    */
   [[nodiscard]] ConstViewType
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
    * \brief Tests whether a vertex is active (present in the mask).
    * \param vertex The vertex index to test.
    * \return \c true if the vertex is in the mask.
    */
   [[nodiscard]] bool
   isActive( IndexType vertex ) const;

   /**
    * \brief Tests whether an edge exists in the induced subgraph.
    * \param source The source vertex index.
    * \param target The target vertex index.
    * \param weight The edge weight.
    * \return \c true if both endpoints are in the mask and the edge passes the edge filter.
    */
   [[nodiscard]] __cuda_callable__
   bool
   edgeExists( IndexType source, IndexType target, const ValueType& weight ) const;

   //! \brief Returns the edge filter.
   [[nodiscard]] __cuda_callable__
   const EdgeFilter&
   getEdgeFilter() const;

   //! \brief Returns the owned mask array.
   [[nodiscard]] const MaskArray&
   getMask() const;

   /**
    * \brief Materializes this MaskedSubGraph into a standalone \ref Graph.
    *
    * Delegates to \c getView().materialize() — the resulting \ref Graph owns
    * its adjacency matrix and contains only the vertices and edges that pass
    * the mask (vertex filter) and the edge filter.
    *
    * \see SubGraph::materialize
    */
   [[nodiscard]] Graph< ValueType, DeviceType, IndexType, GraphOrientation >
   materialize() const;

protected:
   ConstGraphView graphView_;
   MaskArray mask_;
   EdgeFilter edgeFilter_;
};

// ---------------------------------------------------------------------------
// Factory function declarations (indexed)
// ---------------------------------------------------------------------------

/**
 * \brief Creates an MaskedSubGraph from a graph and a vertex index array.
 *
 * \param graph The input graph.
 * \param vertexIndexes Array of vertex indexes that should be active.
 * \return An MaskedSubGraph with all edges accepted.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename Enable = std::enable_if_t< IsArrayType< std::decay_t< VertexIndexes > >::value > >
auto
makeSubGraph( const Graph& graph, const VertexIndexes& vertexIndexes );

/**
 * \brief Creates an MaskedSubGraph from a graph, a vertex index array, and an edge filter.
 *
 * \param graph The input graph.
 * \param vertexIndexes Array of vertex indexes that should be active.
 * \param edgeFilter Callable `(IndexType, IndexType, ValueType) -> bool` selecting edges.
 * \return An MaskedSubGraph with the given mask and edge filter.
 */
template<
   typename Graph,
   typename VertexIndexes,
   typename EdgeFilter,
   typename Enable = std::enable_if_t< IsArrayType< std::decay_t< VertexIndexes > >::value > >
auto
makeSubGraph( const Graph& graph, const VertexIndexes& vertexIndexes, EdgeFilter&& edgeFilter );

/**
 * \brief Free-function adapter: materializes a MaskedSubGraph into a standalone Graph.
 *
 * Equivalent to \c msg.materialize(). Provided for convenience and symmetry
 * with \ref makeSubGraph.
 *
 * \see MaskedSubGraph::materialize
 */
template< typename Graph_, typename EdgeFilter_ >
auto
materialize( const MaskedSubGraph< Graph_, EdgeFilter_ >& msg );

}  // namespace TNL::Graphs

#include "MaskedSubGraph.hpp"
