// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/traverse.h>

#include "breadthFirstSearch.h"
#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "stronglyConnectedComponents.h"

namespace TNL::Graphs::Algorithms {

/**
 * \brief Builds an owning graph containing the edges of \e graph which survive its filters.
 *
 * The edges are iterated with forAllEdges, which applies the vertex and edge
 * filters for SubGraph inputs.  This avoids needing an owning copy of the
 * adjacency matrix for getTransposition (which fails on views).
 *
 * \tparam Reverse  If \c true, the edges are added in the reverse direction.
 */
template< typename Graph >
using FilteredGraph = TNL::Graphs::
   Graph< typename Graph::ValueType, typename Graph::DeviceType, typename Graph::IndexType, typename Graph::GraphOrientation >;

template< bool Reverse, typename Graph >
FilteredGraph< Graph >
buildFilteredGraph( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using OwningGraph = FilteredGraph< Graph >;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   IndexVector capacities( verticesCount, 0 );
   auto capacitiesView = capacities.getView();
   forAllEdges(
      graph,
      [ = ] __cuda_callable__( IndexType src, IndexType, IndexType tgt, const ValueType& ) mutable
      {
         if( tgt == Matrices::paddingIndex< IndexType > )
            return;
         const IndexType row = Reverse ? tgt : src;
         TNL::Algorithms::AtomicOperations< DeviceType >::add( capacitiesView[ row ], 1 );
      },
      launchConfig );

   OwningGraph filteredGraph( verticesCount );
   filteredGraph.setEdgeCounts( capacities );

   IndexVector slots( verticesCount, 0 );
   auto slotView = slots.getView();
   auto matrixView = filteredGraph.getAdjacencyMatrix().getView();
   forAllEdges(
      graph,
      [ = ] __cuda_callable__( IndexType src, IndexType, IndexType tgt, const ValueType& w ) mutable
      {
         if( tgt == Matrices::paddingIndex< IndexType > )
            return;
         const IndexType rowIdx = Reverse ? tgt : src;
         const IndexType column = Reverse ? src : tgt;
         auto row = matrixView.getRow( rowIdx );
         const IndexType idx = TNL::Algorithms::AtomicOperations< DeviceType >::add( slotView[ rowIdx ], 1 );
         row.setElement( idx, column, w );
      },
      launchConfig );

   return filteredGraph;
}

//! \brief Vertex filter accepting the vertices which are not assigned to any component yet.
template< typename ComponentsView >
struct UnassignedVertexFilter
{
   ComponentsView components;

   [[nodiscard]] __cuda_callable__
   bool
   operator()( typename ComponentsView::IndexType vertex ) const
   {
      return components[ vertex ] == 0;
   }
};

/**
 * \brief SCC implementation operating on any graph-like type.
 *
 * Pivot-based SCC: in each round, pick an unassigned vertex as pivot, run
 * forward BFS on the original graph and backward BFS on the reverse graph.
 * Vertices reachable in BOTH directions form one strongly connected component.
 *
 * Both BFS runs are restricted to the unassigned vertices: every path between
 * two vertices of one SCC stays inside that SCC, so the vertices assigned in
 * the previous rounds cannot contribute to the current component.
 *
 * \tparam Graph     Graph, SubGraph, MaskedSubGraph, or GraphView.
 * \tparam Vector    Output vector type for component labels.
 * \tparam VertexPredicate  Unary callable `(Index) -> bool` (vertex filter).
 */
template< typename Graph, typename Vector, typename VertexPredicate >
void
stronglyConnectedComponents_impl(
   const Graph& graph,
   Vector& components,
   VertexPredicate&& vertexPredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( Graph::isDirected(), "SCC requires a directed graph." );
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   components.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   auto componentsView = components.getView();
   auto vertexPredicateCopy = vertexPredicate;
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         componentsView[ vertex ] = vertexPredicateCopy( vertex ) ? 0 : static_cast< IndexType >( -1 );
      } );

   // The filter of the unassigned vertices.  Inactive vertices are marked by -1,
   // so they are excluded as well.  A named functor keeps the SubGraph types
   // short, a lambda here makes the device compilation several times slower.
   const UnassignedVertexFilter< decltype( componentsView ) > unassigned{ componentsView };

   // The reverse graph contains only the edges surviving the filters of the input graph.
   const auto reverseGraph = buildFilteredGraph< true >( graph, launchConfig );
   auto reverseSubGraph = makeSubGraph( reverseGraph, unassigned );

   // A SubGraph cannot be filtered again, so its surviving edges are copied as well.
   const auto forwardGraph = [ & ]
   {
      if constexpr( detail::IsSubGraph< Graph >::value )
         return buildFilteredGraph< false >( graph, launchConfig );
      else
         return graph.getConstView();
   }();
   auto forwardSubGraph = makeSubGraph( forwardGraph, unassigned );

   Vector forwardReachability( verticesCount );
   Vector reverseReachability( verticesCount );

   IndexType componentLabel = 1;
   while( true ) {
      const IndexType pivot = TNL::Algorithms::reduce< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) -> IndexType
         {
            return componentsView[ vertex ] == 0 ? vertex : static_cast< IndexType >( -1 );
         },
         TNL::Max{} );

      if( pivot < 0 )
         return;

      breadthFirstSearch( forwardSubGraph, pivot, forwardReachability, 0.0, 0.0, BFSTopDownMode::Compact, launchConfig );
      breadthFirstSearch( reverseSubGraph, pivot, reverseReachability, 0.0, 0.0, BFSTopDownMode::Compact, launchConfig );

      const auto forwardReachabilityView = forwardReachability.getConstView();
      const auto reverseReachabilityView = reverseReachability.getConstView();
      const IndexType currentLabel = componentLabel;

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( componentsView[ vertex ] == 0 && forwardReachabilityView[ vertex ] >= 0
                && reverseReachabilityView[ vertex ] >= 0 )
               componentsView[ vertex ] = currentLabel;
         } );

      componentLabel++;
   }
}

template< typename Graph, typename Vector >
void
stronglyConnectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( Graph::isDirected(), "SCC requires a directed graph." );
   const auto graphView = graph.getConstView();
   stronglyConnectedComponents_impl(
      graph,
      components,
      [ = ] __cuda_callable__( typename Graph::IndexType vertex )
      {
         return graphView.vertexExists( vertex );
      },
      launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
