// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Graphs/Graph.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/reduce.hpp>

namespace TNL::Benchmarks::Graphs {

template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void
semiringSSSP( const Graph& graph, Index start, Vector& distances )
{
   using Real = typename Graph::ValueType;
   const Index n = graph.getVertexCount();
   typename Graph::AdjacencyMatrixType transposedAdjacencyMatrix;
   transposedAdjacencyMatrix.getTransposition( graph.getAdjacencyMatrix() );

   distances.setSize( n );
   Vector y( distances.getSize() );
   y = distances;

   for( Index i = 1; i <= n; i++ ) {
      auto x_view = distances.getView();
      auto y_view = y.getView();

      auto fetch = [ = ] __cuda_callable__( Index rowIdx, Index columnIdx, const Real& value ) -> Real
      {
         TNL_ASSERT_GE( columnIdx, 0, "" );
         TNL_ASSERT_LT( columnIdx, x_view.getSize(), "" );
         return x_view[ columnIdx ] + value;
      };
      auto store = [ = ] __cuda_callable__( Index rowIdx, const Real& value ) mutable
      {
         TNL_ASSERT_GE( rowIdx, 0, "" );
         TNL_ASSERT_LT( rowIdx, y_view.getSize(), "" );
         TNL_ASSERT_LT( rowIdx, x_view.getSize(), "" );
         y_view[ rowIdx ] = min( x_view[ rowIdx ], value );
      };
      Matrices::reduceAllRows( transposedAdjacencyMatrix, fetch, TNL::Min{}, store );
      if( distances == y )
         break;
      distances = y;
   }
}

}  // namespace TNL::Benchmarks::Graphs
