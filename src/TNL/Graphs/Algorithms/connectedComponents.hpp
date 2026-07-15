// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Atomic.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Matrices/MatrixBase.h>

#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "connectedComponents.h"

namespace TNL::Graphs::Algorithms {

template< typename Index, typename VisitedView, typename ComponentsView, typename GraphView >
void
enqueueComponentVertex(
   const Index componentLabel,
   const Index vertex,
   VisitedView visited,
   ComponentsView components,
   std::queue< Index >& queue,
   const GraphView& graphView )
{
   if( ! graphView.isActive( vertex ) )
      return;
   if( visited[ vertex ] )
      return;

   visited[ vertex ] = 1;
   components[ vertex ] = componentLabel;
   queue.push( vertex );
}

template< typename Graph, typename Vector >
void
connectedComponentsSequential( const Graph& graph, Vector& components )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;

   const IndexType verticesCount = graph.getVertexCount();
   components.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   const auto graphView = graph.getConstView();
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   IndexVector visited( verticesCount, 0 );
   auto visitedView = visited.getView();
   auto componentsView = components.getView();
   components = static_cast< IndexType >( -1 );

   // BFS from each unvisited active vertex.  CC treats the graph as
   // undirected, so we must traverse both outgoing and incoming edges.
   for( IndexType componentLabel = 0; componentLabel < verticesCount; componentLabel++ ) {
      if( ! graphView.isActive( componentLabel ) )
         continue;
      if( visitedView[ componentLabel ] )
         continue;

      std::queue< IndexType > queue;
      enqueueComponentVertex( componentLabel, componentLabel, visitedView, componentsView, queue, graphView );

      while( ! queue.empty() ) {
         const IndexType currentVertex = queue.front();
         queue.pop();

         // Forward edges: currentVertex -> neighbor
         const auto currentRow = adjacencyMatrix.getRow( currentVertex );
         for( IndexType localIdx = 0; localIdx < currentRow.getSize(); localIdx++ ) {
            const IndexType neighbor = currentRow.getColumnIndex( localIdx );
            if( neighbor == Matrices::paddingIndex< IndexType > )
               continue;
            const ValueType weight = currentRow.getValue( localIdx );
            if( ! graphView.edgeExists( currentVertex, neighbor, weight ) )
               continue;
            enqueueComponentVertex( componentLabel, neighbor, visitedView, componentsView, queue, graphView );
         }

         // Reverse edges: rowIdx -> currentVertex.
         // Symmetric matrices store only the lower triangle, so getRow(current)
         // misses neighbors j > current.  We must scan all rows to find edges
         // pointing TO the current vertex (O(n) per dequeue).  Non-symmetric
         // matrices already store both directions, so this scan is skipped.
         if constexpr( AdjacencyMatrixType::isSymmetric() ) {
            for( IndexType rowIdx = 0; rowIdx < verticesCount; rowIdx++ ) {
               if( rowIdx == currentVertex )
                  continue;

               const auto row = adjacencyMatrix.getRow( rowIdx );
               for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
                  const IndexType target = row.getColumnIndex( localIdx );
                  if( target == Matrices::paddingIndex< IndexType > )
                     continue;
                  if( target == currentVertex ) {
                     const ValueType weight = row.getValue( localIdx );
                     if( ! graphView.edgeExists( rowIdx, currentVertex, weight ) )
                        break;
                     enqueueComponentVertex( componentLabel, rowIdx, visitedView, componentsView, queue, graphView );
                     break;
                  }
               }
            }
         }
      }
   }
}

template< typename Graph, typename Vector >
void
connectedComponentsParallel(
   const Graph& graph,
   Vector& components,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   components.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   const auto graphView = graph.getConstView();
   Vector previous( verticesCount );
   Vector relaxed( verticesCount );
   using HostAtomicIndexVector = Containers::Vector< Atomic< IndexType, Devices::Host >, Devices::Host, IndexType >;
   HostAtomicIndexVector hostAtomicComponents;

   if constexpr( std::is_same_v< DeviceType, Devices::Host > )
      hostAtomicComponents.setSize( verticesCount );

   components.forAllElements(
      [ = ] __cuda_callable__( IndexType vertex, IndexType & value )
      {
         value = graphView.isActive( vertex ) ? vertex : static_cast< IndexType >( -1 );
      } );
   previous = static_cast< IndexType >( -1 );

   // Repeatedly propagate the smallest representative across edges until labels stabilize.
   while( previous != components ) {
      previous = components;
      auto componentsView = components.getView();
      const auto previousView = previous.getConstView();
      const IndexType paddingIndex = Matrices::paddingIndex< IndexType >;

      if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
         auto hostAtomicComponentsView = hostAtomicComponents.getView();

         TNL::Algorithms::parallelFor< DeviceType >(
            0,
            verticesCount,
            [ = ] __cuda_callable__( IndexType vertex ) mutable
            {
               hostAtomicComponentsView[ vertex ] = previousView[ vertex ];
            } );

         forAllEdges(
            graph,
            [ = ] __cuda_callable__(
               IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               if( targetIdx == paddingIndex )
                  return;

               const IndexType sourceLabel = previousView[ sourceIdx ];
               const IndexType targetLabel = previousView[ targetIdx ];
               if( sourceLabel == static_cast< IndexType >( -1 ) || targetLabel == static_cast< IndexType >( -1 ) )
                  return;
               const IndexType componentLabel = sourceLabel < targetLabel ? sourceLabel : targetLabel;

               hostAtomicComponentsView[ sourceIdx ].fetch_min( componentLabel );
               hostAtomicComponentsView[ targetIdx ].fetch_min( componentLabel );
            },
            launchConfig );

         TNL::Algorithms::parallelFor< DeviceType >(
            0,
            verticesCount,
            [ = ] __cuda_callable__( IndexType vertex ) mutable
            {
               componentsView[ vertex ] = hostAtomicComponentsView[ vertex ].load();
            } );
      }
      else {
         forAllEdges(
            graph,
            [ = ] __cuda_callable__(
               IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               if( targetIdx == paddingIndex )
                  return;

               const IndexType sourceLabel = previousView[ sourceIdx ];
               const IndexType targetLabel = previousView[ targetIdx ];
               if( sourceLabel == static_cast< IndexType >( -1 ) || targetLabel == static_cast< IndexType >( -1 ) )
                  return;
               const IndexType componentLabel = sourceLabel < targetLabel ? sourceLabel : targetLabel;

               atomicMin( &componentsView[ sourceIdx ], componentLabel );
               atomicMin( &componentsView[ targetIdx ], componentLabel );
            },
            launchConfig );
      }

      relaxed = components;
      const auto relaxedView = relaxed.getConstView();

      // Pointer jumping compresses the representative chains after each relaxation round.
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( relaxedView[ vertex ] != static_cast< IndexType >( -1 ) )
               componentsView[ vertex ] = relaxedView[ relaxedView[ vertex ] ];
         } );
   }
}

template< typename Graph, typename Vector >
void
connectedComponents_impl(
   const Graph& graph,
   Vector& components,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using DeviceType = typename Graph::DeviceType;

   if constexpr( std::is_same_v< DeviceType, Devices::Sequential > || std::is_same_v< DeviceType, Devices::Host > )
      connectedComponentsSequential( graph, components );
   else
      connectedComponentsParallel( graph, components, launchConfig );
}

template< typename Graph, typename Vector >
void
connectedComponents( const Graph& graph, Vector& components, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   connectedComponents_impl( graph, components, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
