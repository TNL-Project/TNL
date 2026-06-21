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

template< typename Graph, typename Vector, typename IsActive, typename EdgePredicate >
void
stronglyConnectedComponents_impl(
   const Graph& graph,
   Vector& components,
   IsActive&& isActive,
   EdgePredicate&& edgePredicate,
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
   auto isActiveCopy = isActive;
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         componentsView[ vertex ] = isActiveCopy( vertex ) ? 0 : static_cast< IndexType >( -1 );
      } );

   // Build the reverse (transposed) adjacency matrix once up front; it is
   // reused in every iteration below.
   typename Graph::AdjacencyMatrixType reverseAdjacencyMatrix;
   reverseAdjacencyMatrix.getTransposition( graph.getAdjacencyMatrix() );
   Graph reverseGraph( std::move( reverseAdjacencyMatrix ) );

   Vector forwardReachability( verticesCount );
   Vector reverseReachability( verticesCount );

   // Pivot-based SCC: in each round we pick any still-unassigned vertex as
   // the pivot, run a forward BFS on the original graph and a backward BFS
   // on the transposed graph.  Vertices reachable in BOTH directions form
   // exactly one strongly connected component.
   IndexType componentLabel = 1;
   while( true ) {
      // NOTE: Finding the pivot via reduce is O(n) per SCC iteration.
      // For graphs with many small SCCs this leads to O(n^2) total work.
      // Pick the largest-indexed unassigned vertex as the next pivot.
      const IndexType pivot = TNL::Algorithms::reduce< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) -> IndexType
         {
            return componentsView[ vertex ] == 0 ? vertex : static_cast< IndexType >( -1 );
         },
         TNL::Max{} );

      if( pivot < 0 )
         return;  // all vertices assigned

      breadthFirstSearchIf( graph, pivot, isActive, edgePredicate, forwardReachability, launchConfig );

      // For the reverse graph, the edge predicate is called with the natural
      // orientation of each stored edge in the transposed matrix, which
      // corresponds to (target, source, weight) of the original graph.
      breadthFirstSearchIf( reverseGraph, pivot, isActive, edgePredicate, reverseReachability, launchConfig );

      const auto forwardReachabilityView = forwardReachability.getConstView();
      const auto reverseReachabilityView = reverseReachability.getConstView();
      const IndexType currentLabel = componentLabel;

      // A vertex belongs to this SCC iff it is reachable from the pivot in
      // both the forward and the reverse direction.
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
   using IndexType = typename Graph::IndexType;
   stronglyConnectedComponents_impl(
      graph,
      components,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename Enable >
void
stronglyConnectedComponents(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
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
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Enable >
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
   stronglyConnectedComponents_impl(
      graph,
      components,
      isActive,
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename Enable >
void
stronglyConnectedComponents(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
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
   stronglyConnectedComponents_impl(
      graph, components, isActive, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
stronglyConnectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   stronglyConnectedComponents_impl(
      graph,
      components,
      predicate,
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
stronglyConnectedComponentsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto vPredicate = std::forward< VertexPredicate >( vertexPredicate );
   stronglyConnectedComponents_impl(
      graph, components, vPredicate, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
