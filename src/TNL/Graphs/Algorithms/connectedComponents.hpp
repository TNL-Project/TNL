// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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

template< typename Index, typename ParentView >
Index
findComponentRoot( ParentView parent, Index vertex )
{
   // Path halving keeps the trees shallow without recursion.
   while( parent[ vertex ] != vertex ) {
      parent[ vertex ] = parent[ parent[ vertex ] ];
      vertex = parent[ vertex ];
   }
   return vertex;
}

template< typename Graph, typename Vector >
void
connectedComponentsSequential( const Graph& graph, Vector& components )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;

   const IndexType verticesCount = graph.getVertexCount();
   components.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   const auto graphView = graph.getConstView();
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   auto componentsView = components.getView();
   for( IndexType vertex = 0; vertex < verticesCount; vertex++ )
      componentsView[ vertex ] = graphView.vertexExists( vertex ) ? vertex : static_cast< IndexType >( -1 );

   // Union-find over the stored edges.  CC treats the graph as undirected, so each
   // stored edge joins both endpoints regardless of its direction.  This also covers
   // symmetric matrices, which store only the lower triangle.  The smaller root
   // always becomes the parent, so each root is the smallest vertex of its component.
   for( IndexType sourceIdx = 0; sourceIdx < verticesCount; sourceIdx++ ) {
      if( componentsView[ sourceIdx ] == static_cast< IndexType >( -1 ) )
         continue;
      const auto row = adjacencyMatrix.getRow( sourceIdx );
      for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
         const IndexType targetIdx = row.getColumnIndex( localIdx );
         if( targetIdx == Matrices::paddingIndex< IndexType > )
            continue;
         if( componentsView[ targetIdx ] == static_cast< IndexType >( -1 ) )
            continue;
         const ValueType weight = row.getValue( localIdx );
         if( ! graphView.edgeExists( sourceIdx, targetIdx, weight ) )
            continue;
         const IndexType sourceRoot = findComponentRoot( componentsView, sourceIdx );
         const IndexType targetRoot = findComponentRoot( componentsView, targetIdx );
         if( sourceRoot < targetRoot )
            componentsView[ targetRoot ] = sourceRoot;
         else if( targetRoot < sourceRoot )
            componentsView[ sourceRoot ] = targetRoot;
      }
   }

   for( IndexType vertex = 0; vertex < verticesCount; vertex++ )
      if( componentsView[ vertex ] != static_cast< IndexType >( -1 ) )
         componentsView[ vertex ] = findComponentRoot( componentsView, vertex );
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
      [ = ] __cuda_callable__( IndexType vertex, IndexType& value )
      {
         value = graphView.vertexExists( vertex ) ? vertex : static_cast< IndexType >( -1 );
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
