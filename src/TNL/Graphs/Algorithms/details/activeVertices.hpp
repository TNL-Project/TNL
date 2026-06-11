// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>

namespace TNL::Graphs::Algorithms::detail {

template< typename Graph, typename ActiveVector >
void
activateAllVertices( const Graph& graph, ActiveVector& activeVertices )
{
   using IndexType = typename Graph::IndexType;

   activeVertices.setSize( graph.getVertexCount() );
   activeVertices = static_cast< IndexType >( 1 );
}

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

   const auto vertexIndexes_view = vertexIndexes.getConstView();
   auto activeVertices_view = activeVertices.getView();
   const bool validIndexes = TNL::Algorithms::reduce< DeviceType >(
      0,
      selectedVerticesCount,
      [ = ] __cuda_callable__( IndexType index ) mutable -> bool
      {
         const IndexType vertex = vertexIndexes_view[ index ];
         if( vertex >= 0 && vertex < verticesCount ) {
            activeVertices_view[ vertex ] = static_cast< IndexType >( 1 );
            return true;
         }
         else
            return false;
      },
      TNL::LogicalAnd{} );

   if( ! validIndexes )
      throw std::invalid_argument( "Vertex indexes for maximalIndependentSet must be valid graph vertices." );
}

template< typename Graph, typename VertexPredicate, typename ActiveVector >
void
activateVerticesIf( const Graph& graph, VertexPredicate&& vertexPredicate, ActiveVector& activeVertices )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );

   activeVertices.setSize( verticesCount );
   auto activeVerticesView = activeVertices.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         activeVerticesView[ vertex ] = predicate( vertex ) ? static_cast< IndexType >( 1 ) : static_cast< IndexType >( 0 );
      } );
}

}  // namespace TNL::Graphs::Algorithms::detail
