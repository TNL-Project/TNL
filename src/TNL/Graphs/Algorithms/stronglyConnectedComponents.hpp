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
 * \brief SCC implementation operating on any graph-like type.
 *
 * Pivot-based SCC: in each round, pick an unassigned vertex as pivot, run
 * forward BFS on the original graph and backward BFS on the reverse graph.
 * Vertices reachable in BOTH directions form one strongly connected component.
 *
 * The reverse graph is built by iterating the forward graph's edges with
 * forAllEdges (which applies vertex and edge filters for SubGraph inputs) and
 * adding them in reverse direction.  This avoids needing an owning copy of
 * the adjacency matrix for getTransposition (which fails on views).
 *
 * \tparam Graph     Graph, SubGraph, MaskedSubGraph, or GraphView.
 * \tparam Vector    Output vector type for component labels.
 * \tparam IsActive  Unary callable `(Index) -> bool` (vertex filter).
 */
template< typename Graph, typename Vector, typename IsActive >
void
stronglyConnectedComponents_impl(
   const Graph& graph,
   Vector& components,
   IsActive&& isActive,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( Graph::isDirected(), "SCC requires a directed graph." );
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using GraphOrientation = typename Graph::GraphOrientation;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;

   components.setSize( verticesCount );

   auto componentsView = components.getView();
   auto isActiveCopy = isActive;
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         componentsView[ vertex ] = isActiveCopy( vertex ) ? 0 : static_cast< IndexType >( -1 );
      } );

   // Build the reverse graph by iterating the forward graph's edges.
   // For SubGraph inputs, forAllEdges applies vertex and edge filters
   // transparently, so only surviving edges are added to the reverse graph.
   using OwningGraph = TNL::Graphs::Graph< ValueType, DeviceType, IndexType, GraphOrientation >;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector reverseCapacities( verticesCount, 0 );
   auto revCapView = reverseCapacities.getView();
   forAllEdges(
      graph,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType tgt, const ValueType& ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( revCapView[ tgt ], 1 );
      },
      launchConfig );

   OwningGraph reverseGraph( verticesCount );
   reverseGraph.setEdgeCounts( reverseCapacities );

   IndexVector slots( verticesCount, 0 );
   auto slotView = slots.getView();
   auto revMatrixView = reverseGraph.getAdjacencyMatrix().getView();
   forAllEdges(
      graph,
      [ = ] __cuda_callable__( IndexType src, IndexType, IndexType tgt, const ValueType& w ) mutable
      {
         auto row = revMatrixView.getRow( tgt );
         const IndexType idx = TNL::Algorithms::AtomicOperations< DeviceType >::add( slotView[ tgt ], 1 );
         row.setElement( idx, src, w );
      },
      launchConfig );

   // The reverse graph already contains only filtered edges (forAllEdges
   // applied the filters).  The reverse SubGraph only needs the vertex filter
   // to prevent BFS from visiting inactive vertices.
   auto reverseSubGraph = makeSubGraph( reverseGraph, isActive );

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

      breadthFirstSearch( graph, pivot, forwardReachability, launchConfig );
      breadthFirstSearch( reverseSubGraph, pivot, reverseReachability, launchConfig );

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
         return graphView.isActive( vertex );
      },
      launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
