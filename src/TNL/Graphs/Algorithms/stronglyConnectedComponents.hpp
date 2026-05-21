// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>

#include "breadthFirstSearch.h"
#include "stronglyConnectedComponents.h"

namespace TNL::Graphs::Algorithms {

template< typename Graph, typename Vector >
void
stronglyConnectedComponents(
   const Graph& graph,
   Vector& components,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( Graph::isDirected(), "SCC requires a directed graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;

   components.setSize( verticesCount );

   components = 0;
   auto componentsView = components.getView();

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
            return componentsView[ vertex ] == 0 ? vertex : -1;
         },
         TNL::Max{} );

      if( pivot < 0 )
         return;

      // A pivot SCC is exactly the intersection of forward and reverse reachability.
      breadthFirstSearch( graph, pivot, forwardReachability, launchConfig );
      breadthFirstSearch( reverseGraph, pivot, reverseReachability, launchConfig );

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

}  // namespace TNL::Graphs::Algorithms
