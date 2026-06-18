// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>

namespace TNL::Graphs::Algorithms::detail {

template< typename Graph, typename VertexIndexes, typename ActiveVector >
void
activateIndexedVertices( const Graph& graph, const VertexIndexes& vertexIndexes, ActiveVector& activeVertices )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   const IndexType selectedVerticesCount = static_cast< IndexType >( vertexIndexes.getSize() );

   activeVertices.setSize( verticesCount );
   activeVertices = static_cast< IndexType >( 0 );

   if( selectedVerticesCount == 0 )
      return;

   const auto vertexIndexesView = vertexIndexes.getConstView();
   auto activeVerticesView = activeVertices.getView();
   const bool validIndexes = TNL::Algorithms::reduce< DeviceType >(
      0,
      selectedVerticesCount,
      [ = ] __cuda_callable__( IndexType index ) mutable -> bool
      {
         const IndexType vertex = vertexIndexesView[ index ];
         if( vertex >= 0 && vertex < verticesCount ) {
            activeVerticesView[ vertex ] = static_cast< IndexType >( 1 );
            return true;
         }
         else
            return false;
      },
      TNL::LogicalAnd{} );

   if( ! validIndexes )
      throw std::invalid_argument( "Vertex indexes must be valid graph vertices." );
}

}  // namespace TNL::Graphs::Algorithms::detail
