// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/reduce.hpp>

namespace TNL::Benchmarks::Graphs {

template< typename Graph, typename Vector >
void
semiringBFS( const Graph& graph, typename Graph::IndexType start, Vector& distances )
{
   using Real = typename Graph::ValueType;
   using Index = typename Graph::IndexType;
   const Index n = graph.getVertexCount();
   typename Graph::AdjacencyMatrixType transposedAdjacencyMatrix;
   transposedAdjacencyMatrix.getTransposition( graph.getAdjacencyMatrix() );
   distances.setSize( n );

   Vector y( distances.getSize() );
   distances = 0;
   distances.setElement( start, 1 );
   y = distances;

   for( Index i = 2; i <= n; i++ ) {
      auto x_view = distances.getView();
      auto y_view = y.getView();

      auto fetch = [ = ] __cuda_callable__( Index rowIdx, Index columnIdx, const Real& value ) -> bool
      {
         return x_view[ columnIdx ] && value;
      };
      auto store = [ = ] __cuda_callable__( Index rowIdx, const Real& value ) mutable
      {
         if( value && x_view[ rowIdx ] == 0 ) {
            y_view[ rowIdx ] = i;
         }
      };
      Matrices::reduceAllRows( transposedAdjacencyMatrix, fetch, TNL::Plus{}, store );
      if( distances == y )
         break;
      distances = y;
   }
   distances -= 1;
}

}  // namespace TNL::Benchmarks::Graphs
