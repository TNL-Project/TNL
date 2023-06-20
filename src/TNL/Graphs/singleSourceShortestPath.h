// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Graphs/Graph.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>

namespace TNL::Graphs {

template< typename Matrix, typename Vector, typename Index = typename Matrix::IndexType >
void singleSourceShortestPathTransposed( const Matrix& transposedAdjacencyMatrix, Index start, Vector& distances )
{
   TNL_ASSERT_TRUE( transposedAdjacencyMatrix.getRows() == transposedAdjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == transposedAdjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   const Index n = transposedAdjacencyMatrix.getRows();

   Vector y( distances.getSize() );
   y = distances;

   for( Index i = 1; i <= n; i++ )
   {
      auto x_view = distances.getView();
      auto y_view = y.getView();

      auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const Real& value ) -> Real {
         return x_view[ columnIdx ] + value;
      };
      auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
            y_view[ rowIdx ] = min( x_view[ rowIdx ], value );
      };
      transposedAdjacencyMatrix.reduceAllRows( fetch, TNL::Min{}, keep, std::numeric_limits< Real >::max() );
      if( distances == y )
         break;
      distances = y;
   }
}

template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void singleSourceShortestPath( const Graph& graph, Index start, Vector& distances )
{
   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;

   distances.setSize( graph.getNodeCount() );
   distances = std::numeric_limits< Real >::max();
   distances.setElement( start, 0.0 );

   // In the sequential version, we use the Dijkstra algorithm.
   if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value )
   {

      // The priority queue stores pairs of (distance, vertex)
      std::priority_queue< std::pair< Real, Index >, std::vector< std::pair< Real, Index > >, std::greater< std::pair< Real, Index >>> pq;
      pq.emplace(0, start);

      while( !pq.empty() ) {
         Real current_distance;
         Index current;
         std::tie(current_distance, current) = pq.top();
         pq.pop();

         if (current_distance > distances[current]) {
            continue;
         }

         const auto row = graph.getAdjacencyMatrix().getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& edge_weight = row.getValue( i );
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == graph.getAdjacencyMatrix().getPaddingIndex() )
               continue;
            double distance = current_distance + edge_weight;

            if( distance < distances[ neighbor ] ) {
               distances[neighbor] = distance;
               pq.emplace(distance, neighbor);
            }
         }
      }
   }
   else
   {
      typename Graph::MatrixType transposed;
      transposed.transpose( graph.getAdjacencyMatrix() );
      singleSourceShortestPathTransposed( transposed, start, distances );
   }
   distances.forAllElements( [] __cuda_callable__ ( Index i, Real& x ) {
      x = ( x == std::numeric_limits< Real >::max() ) ? -1.0 : x; }
   );
}

}  // namespace TNL::Graphs
