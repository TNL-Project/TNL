// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MaskedSubGraph.h"

#include <TNL/Graphs/Algorithms/details/activeVertices.hpp>

namespace TNL::Graphs {

// ---------------------------------------------------------------------------
// MaskedSubGraph static methods
// ---------------------------------------------------------------------------

template< typename Graph_, typename EdgeFilter >
constexpr bool
MaskedSubGraph< Graph_, EdgeFilter >::isDirected()
{
   return GraphType::isDirected();
}

template< typename Graph_, typename EdgeFilter >
constexpr bool
MaskedSubGraph< Graph_, EdgeFilter >::isUndirected()
{
   return GraphType::isUndirected();
}

// ---------------------------------------------------------------------------
// MaskedSubGraph constructor
// ---------------------------------------------------------------------------

template< typename Graph_, typename EdgeFilter >
template< typename VertexIndexes >
MaskedSubGraph< Graph_, EdgeFilter >::MaskedSubGraph(
   const GraphType& graph,
   const VertexIndexes& vertexIndexes,
   EdgeFilter edgeFilter )
: graphView_( graph.getConstView() ),
  mask_(),
  edgeFilter_( std::move( edgeFilter ) )
{
   Algorithms::detail::activateIndexedVertices( graph, vertexIndexes, mask_ );
}

// ---------------------------------------------------------------------------
// MaskedSubGraph methods
// ---------------------------------------------------------------------------

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
MaskedSubGraph< Graph_, EdgeFilter >::getVertexCount() const -> IndexType
{
   return graphView_.getVertexCount();
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] auto
MaskedSubGraph< Graph_, EdgeFilter >::getEdgeCount() const -> IndexType
{
   return graphView_.getEdgeCount();
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
MaskedSubGraph< Graph_, EdgeFilter >::getAdjacencyMatrixView() const -> const ConstAdjacencyMatrixView&
{
   return graphView_.getAdjacencyMatrixView();
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
MaskedSubGraph< Graph_, EdgeFilter >::getAdjacencyMatrix() const -> const AdjacencyMatrixType&
{
   return graphView_.getAdjacencyMatrixView();
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] auto
MaskedSubGraph< Graph_, EdgeFilter >::getView() -> ViewType
{
   return ViewType( graphView_, VertexFilter{ mask_.getConstView() }, edgeFilter_ );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] auto
MaskedSubGraph< Graph_, EdgeFilter >::getConstView() const -> ConstViewType
{
   return ConstViewType( graphView_, VertexFilter{ mask_.getConstView() }, edgeFilter_ );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
MaskedSubGraph< Graph_, EdgeFilter >::getVertex( IndexType vertexIdx ) const -> ConstVertexView
{
   return graphView_.getVertex( vertexIdx );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
MaskedSubGraph< Graph_, EdgeFilter >::getVertexDegree( IndexType vertexIdx ) const -> IndexType
{
   return graphView_.getVertexDegree( vertexIdx );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] bool
MaskedSubGraph< Graph_, EdgeFilter >::isActive( IndexType vertex ) const
{
   return static_cast< bool >( mask_.getElement( vertex ) );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
bool
MaskedSubGraph< Graph_, EdgeFilter >::edgeExists( IndexType source, IndexType target, const ValueType& weight ) const
{
   return isActive( source ) && isActive( target ) && edgeFilter_( source, target, weight );
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
const EdgeFilter&
MaskedSubGraph< Graph_, EdgeFilter >::getEdgeFilter() const
{
   return edgeFilter_;
}

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] auto
MaskedSubGraph< Graph_, EdgeFilter >::getMask() const -> const MaskArray&
{
   return mask_;
}

// ---------------------------------------------------------------------------
// Factory functions (indexed)
// ---------------------------------------------------------------------------

template< typename Graph, typename VertexIndexes, typename Enable >
auto
makeSubGraph( const Graph& graph, const VertexIndexes& vertexIndexes )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using AcceptAllEdges = detail::AcceptAllEdges< IndexType, ValueType >;
   return MaskedSubGraph< Graph, AcceptAllEdges >( graph, vertexIndexes, AcceptAllEdges{} );
}

template< typename Graph, typename VertexIndexes, typename EdgeFilter, typename Enable >
auto
makeSubGraph( const Graph& graph, const VertexIndexes& vertexIndexes, EdgeFilter&& edgeFilter )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   static_assert(
      detail::isEdgeFilterCallable_v< std::decay_t< EdgeFilter >, IndexType, ValueType >,
      "Edge filter must be callable as (IndexType, IndexType, ValueType) -> bool." );
   return MaskedSubGraph< Graph, std::decay_t< EdgeFilter > >( graph, vertexIndexes, std::forward< EdgeFilter >( edgeFilter ) );
}

// ---------------------------------------------------------------------------
// MaskedSubGraph::materialize (delegates to the SubGraph view)
// ---------------------------------------------------------------------------

template< typename Graph_, typename EdgeFilter >
[[nodiscard]] Graph<
   typename MaskedSubGraph< Graph_, EdgeFilter >::ValueType,
   typename MaskedSubGraph< Graph_, EdgeFilter >::DeviceType,
   typename MaskedSubGraph< Graph_, EdgeFilter >::IndexType,
   typename MaskedSubGraph< Graph_, EdgeFilter >::GraphOrientation >
MaskedSubGraph< Graph_, EdgeFilter >::materialize() const
{
   return this->getConstView().materialize();
}

// ---------------------------------------------------------------------------
// Free-function adapter: materialize(MaskedSubGraph)
// ---------------------------------------------------------------------------

template< typename Graph_, typename EdgeFilter_ >
auto
materialize( const MaskedSubGraph< Graph_, EdgeFilter_ >& msg )
{
   return msg.materialize();
}

}  // namespace TNL::Graphs
