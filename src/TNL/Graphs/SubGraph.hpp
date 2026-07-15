// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SubGraph.h"

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>

#include "Graph.h"
#include "reduce.h"
#include "traverse.h"

namespace TNL::Graphs {

// ---------------------------------------------------------------------------
// Default filter types
// ---------------------------------------------------------------------------

namespace detail {

/// Default vertex filter: accepts all vertices.
template< typename IndexType >
struct AcceptAllVertices
{
   __cuda_callable__
   bool
   operator()( IndexType ) const
   {
      return true;
   }
};

/// Default edge filter: accepts all edges.
template< typename IndexType, typename ValueType >
struct AcceptAllEdges
{
   __cuda_callable__
   bool
   operator()( IndexType, IndexType, const ValueType& ) const
   {
      return true;
   }
};

/// Vertex filter backed by a dense boolean mask (array view).
template< typename MaskView >
struct MaskVertexFilter
{
   MaskView mask;

   __cuda_callable__
   bool
   operator()( typename MaskView::IndexType vertex ) const
   {
      return static_cast< bool >( mask[ vertex ] );
   }
};

/// Type trait: T is a callable that can serve as a vertex filter.
template< typename T, typename IndexType >
struct IsVertexFilterCallable : std::bool_constant< std::is_invocable_r_v< bool, T, IndexType > >
{};

template< typename T, typename IndexType >
constexpr bool isVertexFilterCallable_v = IsVertexFilterCallable< T, IndexType >::value;

/// Type trait: T is a callable that can serve as an edge filter.
template< typename T, typename IndexType, typename ValueType >
struct IsEdgeFilterCallable : std::bool_constant< std::is_invocable_r_v< bool, T, IndexType, IndexType, ValueType > >
{};

template< typename T, typename IndexType, typename ValueType >
constexpr bool isEdgeFilterCallable_v = IsEdgeFilterCallable< T, IndexType, ValueType >::value;

}  // namespace detail

// ---------------------------------------------------------------------------
// SubGraph static methods
// ---------------------------------------------------------------------------

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
constexpr bool
SubGraph< Graph_, VertexFilter, EdgeFilter >::isDirected()
{
   return GraphType::isDirected();
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
constexpr bool
SubGraph< Graph_, VertexFilter, EdgeFilter >::isUndirected()
{
   return GraphType::isUndirected();
}

// ---------------------------------------------------------------------------
// SubGraph constructor
// ---------------------------------------------------------------------------

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
SubGraph< Graph_, VertexFilter, EdgeFilter >::SubGraph(
   const GraphType& graph,
   VertexFilter vertexFilter,
   EdgeFilter edgeFilter )
: graphView_( graph.getConstView() ),
  vertexFilter_( std::move( vertexFilter ) ),
  edgeFilter_( std::move( edgeFilter ) )
{}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
SubGraph< Graph_, VertexFilter, EdgeFilter >::SubGraph(
   ConstGraphView graphView,
   VertexFilter vertexFilter,
   EdgeFilter edgeFilter )
: graphView_( std::move( graphView ) ),
  vertexFilter_( std::move( vertexFilter ) ),
  edgeFilter_( std::move( edgeFilter ) )
{}

// ---------------------------------------------------------------------------
// SubGraph methods
// ---------------------------------------------------------------------------

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getVertexCount() const -> IndexType
{
   return graphView_.getVertexCount();
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getEdgeCount() const -> IndexType
{
   return graphView_.getEdgeCount();
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getAdjacencyMatrixView() const -> const ConstAdjacencyMatrixView&
{
   return graphView_.getAdjacencyMatrixView();
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getAdjacencyMatrix() const -> const AdjacencyMatrixType&
{
   return graphView_.getAdjacencyMatrixView();
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getView() -> ViewType
{
   return *this;
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getVertex( IndexType vertexIdx ) const -> ConstVertexView
{
   return graphView_.getVertex( vertexIdx );
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getVertexDegree( IndexType vertexIdx ) const -> IndexType
{
   return graphView_.getVertexDegree( vertexIdx );
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
bool
SubGraph< Graph_, VertexFilter, EdgeFilter >::vertexExists( IndexType vertex ) const
{
   return vertexFilter_( vertex );
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
bool
SubGraph< Graph_, VertexFilter, EdgeFilter >::edgeExists( IndexType source, IndexType target, const ValueType& weight ) const
{
   return vertexFilter_( source ) && vertexFilter_( target ) && edgeFilter_( source, target, weight );
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getVertexFilter() const -> const VertexFilter&
{
   return vertexFilter_;
}

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] __cuda_callable__
auto
SubGraph< Graph_, VertexFilter, EdgeFilter >::getEdgeFilter() const -> const EdgeFilter&
{
   return edgeFilter_;
}

// ---------------------------------------------------------------------------
// Factory functions (non-indexed)
// ---------------------------------------------------------------------------

template< typename Graph >
auto
makeSubGraph( const Graph& graph )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   return SubGraph< Graph, detail::AcceptAllVertices< IndexType >, detail::AcceptAllEdges< IndexType, ValueType > >(
      graph, detail::AcceptAllVertices< IndexType >{}, detail::AcceptAllEdges< IndexType, ValueType >{} );
}

template< typename Graph, typename VertexFilter, typename Enable >
auto
makeSubGraph( const Graph& graph, VertexFilter&& vertexFilter )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   static_assert(
      detail::isVertexFilterCallable_v< std::decay_t< VertexFilter >, IndexType >,
      "Vertex filter must be callable as (IndexType) -> bool." );
   return SubGraph< Graph, std::decay_t< VertexFilter >, detail::AcceptAllEdges< IndexType, ValueType > >(
      graph, std::forward< VertexFilter >( vertexFilter ), detail::AcceptAllEdges< IndexType, ValueType >{} );
}

template< typename Graph, typename EdgeFilter >
auto
makeSubGraph( const Graph& graph, EdgeOnlyTag, EdgeFilter&& edgeFilter )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   static_assert(
      detail::isEdgeFilterCallable_v< std::decay_t< EdgeFilter >, IndexType, ValueType >,
      "Edge filter must be callable as (IndexType, IndexType, ValueType) -> bool." );
   return SubGraph< Graph, detail::AcceptAllVertices< IndexType >, std::decay_t< EdgeFilter > >(
      graph, detail::AcceptAllVertices< IndexType >{}, std::forward< EdgeFilter >( edgeFilter ) );
}

template< typename Graph, typename VertexFilter, typename EdgeFilter, typename Enable >
auto
makeSubGraph( const Graph& graph, VertexFilter&& vertexFilter, EdgeFilter&& edgeFilter )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   static_assert(
      detail::isVertexFilterCallable_v< std::decay_t< VertexFilter >, IndexType >,
      "Vertex filter must be callable as (IndexType) -> bool." );
   static_assert(
      detail::isEdgeFilterCallable_v< std::decay_t< EdgeFilter >, IndexType, ValueType >,
      "Edge filter must be callable as (IndexType, IndexType, ValueType) -> bool." );
   return SubGraph< Graph, std::decay_t< VertexFilter >, std::decay_t< EdgeFilter > >(
      graph, std::forward< VertexFilter >( vertexFilter ), std::forward< EdgeFilter >( edgeFilter ) );
}

// ---------------------------------------------------------------------------
// Free-function adapters: vertexExists / edgeExists
// ---------------------------------------------------------------------------

template< typename Graph >
__cuda_callable__
bool
vertexExists( const Graph&, typename Graph::IndexType )
{
   return true;
}

template< typename Graph_, typename VertexFilter_, typename EdgeFilter_ >
__cuda_callable__
bool
vertexExists(
   const SubGraph< Graph_, VertexFilter_, EdgeFilter_ >& sg,
   typename SubGraph< Graph_, VertexFilter_, EdgeFilter_ >::IndexType vertex )
{
   return sg.vertexExists( vertex );
}

template< typename Graph >
__cuda_callable__
bool
edgeExists( const Graph&, typename Graph::IndexType, typename Graph::IndexType, const typename Graph::ValueType& )
{
   return true;
}

template< typename Graph_, typename VertexFilter_, typename EdgeFilter_ >
__cuda_callable__
bool
edgeExists(
   const SubGraph< Graph_, VertexFilter_, EdgeFilter_ >& sg,
   typename SubGraph< Graph_, VertexFilter_, EdgeFilter_ >::IndexType source,
   typename SubGraph< Graph_, VertexFilter_, EdgeFilter_ >::IndexType target,
   const typename SubGraph< Graph_, VertexFilter_, EdgeFilter_ >::ValueType& weight )
{
   return sg.edgeExists( source, target, weight );
}

// ---------------------------------------------------------------------------
// SubGraph::materialize
// ---------------------------------------------------------------------------

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
[[nodiscard]] Graph<
   typename SubGraph< Graph_, VertexFilter, EdgeFilter >::ValueType,
   typename SubGraph< Graph_, VertexFilter, EdgeFilter >::DeviceType,
   typename SubGraph< Graph_, VertexFilter, EdgeFilter >::IndexType,
   typename SubGraph< Graph_, VertexFilter, EdgeFilter >::GraphOrientation >
SubGraph< Graph_, VertexFilter, EdgeFilter >::materialize() const
{
   using ResultGraph = Graph< ValueType, DeviceType, IndexType, GraphOrientation >;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType vertexCount = this->getVertexCount();

   // 1. Compute per-vertex edge counts: each surviving edge contributes 1.
   IndexVector capacities( vertexCount, 0 );
   auto capView = capacities.getView();
   reduceAllVertices(
      *this,
      [] __cuda_callable__( IndexType, IndexType, const ValueType& ) -> IndexType
      {
         return 1;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType row, IndexType val ) mutable
      {
         capView[ row ] = val;
      },
      IndexType( 0 ),
      TNL::Algorithms::Segments::LaunchConfiguration{} );

   // 2. Allocate the result graph with the computed row capacities.
   ResultGraph result( vertexCount );
   result.setEdgeCounts( capacities );

   // 3. Fill the edges using an atomic slot counter per source vertex.
   //    `forAllEdges(*this)` applies the vertex filter (on both endpoints)
   //    and the edge filter transparently, so only the surviving edges
   //    of the induced subgraph are written.
   IndexVector slots( vertexCount, 0 );
   auto slotView = slots.getView();
   auto matrixView = result.getAdjacencyMatrix().getView();
   forAllEdges(
      *this,
      [ = ] __cuda_callable__( IndexType src, IndexType, IndexType tgt, const ValueType& w ) mutable
      {
         auto row = matrixView.getRow( src );
         const IndexType idx = TNL::Algorithms::AtomicOperations< DeviceType >::add( slotView[ src ], 1 );
         row.setElement( idx, tgt, w );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );

   return result;
}

// ---------------------------------------------------------------------------
// Free-function adapter: materialize(SubGraph)
// ---------------------------------------------------------------------------

template< typename Graph_, typename VertexFilter_, typename EdgeFilter_ >
auto
materialize( const SubGraph< Graph_, VertexFilter_, EdgeFilter_ >& sg )
{
   return sg.materialize();
}

}  // namespace TNL::Graphs
