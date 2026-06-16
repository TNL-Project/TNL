// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>

#include "breadthFirstSearch.h"
#include "details/activeVertices.hpp"
#include "stronglyConnectedComponents.h"

namespace TNL::Graphs::Algorithms {

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

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;

   components.setSize( verticesCount );

   // Initialize: active vertices get 0 (unassigned), inactive get -1.
   auto componentsView = components.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ =, isActive = isActive ] __cuda_callable__( IndexType vertex ) mutable
      {
         componentsView[ vertex ] = isActive( vertex ) ? 0 : static_cast< IndexType >( -1 );
      } );

   typename Graph::AdjacencyMatrixType reverseAdjacencyMatrix;
   reverseAdjacencyMatrix.getTransposition( graph.getAdjacencyMatrix() );
   Graph reverseGraph( std::move( reverseAdjacencyMatrix ) );

   Vector forwardReachability( verticesCount );
   Vector reverseReachability( verticesCount );
   IndexType currentComponent = 1;

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

      breadthFirstSearchIf( graph, pivot, isActive, forwardReachability, launchConfig );
      breadthFirstSearchIf( reverseGraph, pivot, isActive, reverseReachability, launchConfig );

      const auto forwardReachabilityView = forwardReachability.getConstView();
      const auto reverseReachabilityView = reverseReachability.getConstView();
      const IndexType componentLabel = currentComponent;

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( componentsView[ vertex ] == 0 && forwardReachabilityView[ vertex ] >= 0
                && reverseReachabilityView[ vertex ] >= 0 )
               componentsView[ vertex ] = componentLabel;
         } );

      currentComponent++;
   }
}

template< typename Graph, typename Vector >
void
stronglyConnectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   stronglyConnectedComponents_impl(
      graph,
      components,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
stronglyConnectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   stronglyConnectedComponents_impl( graph, components, isActive, launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
stronglyConnectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   stronglyConnectedComponents_impl( graph, components, predicate, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
