// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Graphs/Graph.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Atomic.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs {

template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void
parallelSingleSourceShortestPath(
   const Graph& graph,
   Index start,
   Vector& distances,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() )
{
   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   const Index n = graph.getVertexCount();
   distances.setSize( n );

   Vector y( distances.getSize() );
   Containers::Vector< Index, Device, Index > predecesors( n, -1 ), marks( n ), marks_scan( n, 0 ), frontier( n, 0 );
   distances = std::numeric_limits< Real >::max();
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   Index frontier_size( 1 );
   y = distances;
   auto y_view = y.getView();
   auto predecesors_view = predecesors.getView();
   auto marks_view = marks.getView();
   auto marks_scan_view = marks_scan.getView();
   for( Index i = 0; i <= n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< Device, Devices::Host > )
         adjacencyMatrix.forElements(
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
            {
               if( columnIdx != Matrices::paddingIndex< Index > ) {
                  Real new_distance = y_view[ rowIdx ] + value;
                  if( new_distance < y_view[ columnIdx ] ) {
#pragma omp atomic write
                     y_view[ columnIdx ] = new_distance;
#pragma omp atomic write
                     predecesors_view[ columnIdx ] = rowIdx;
#pragma omp atomic write
                     marks_view[ columnIdx ] = 1;
                  }
               }
            },
            launchConfig );
      else
         adjacencyMatrix.forElements(
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
            {
               TNL_ASSERT_GE( rowIdx, 0, "" );
               TNL_ASSERT_LT( rowIdx, y_view.getSize(), "" );
               TNL_ASSERT_GE( columnIdx, 0, "" );
               TNL_ASSERT_LT( columnIdx, y_view.getSize(), "" );
               if( columnIdx != Matrices::paddingIndex< Index > ) {
                  Real new_distance = y_view[ rowIdx ] + value;
                  if( new_distance < y_view[ columnIdx ] ) {
                     atomicMin( &y_view[ columnIdx ], new_distance );
                     atomicMin( &predecesors_view[ columnIdx ], rowIdx );
                     atomicMax( &marks_view[ columnIdx ], 1 );
                  }
               }
            },
            launchConfig );
      Algorithms::inclusiveScan( marks, marks_scan );
      frontier_size = marks_scan.getElement( n - 1 );
      if( frontier_size == 0 )
         break;
      frontier = 0;
      auto frontier_view = frontier.getView();
      auto f = [ = ] __cuda_callable__( const Index idx, const Index value ) mutable
      {
         if( idx == 0 ) {
            if( marks_scan_view[ 0 ] == 1 )
               frontier_view[ 0 ] = idx;
         }
         else if( marks_scan_view[ idx ] - marks_scan_view[ idx - 1 ] == 1 )
            frontier_view[ marks_scan_view[ idx ] - 1 ] = idx;
      };
      marks_scan.forAllElements( f );
      distances = y;
   }
}

template< typename Graph, typename Vector, typename Index = typename Graph::IndexType >
void
singleSourceShortestPath( const Graph& graph,
                          Index start,
                          Vector& distances,
                          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() )
{
   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;

   distances.setSize( graph.getVertexCount() );
   distances = std::numeric_limits< Real >::max();
   distances.setElement( start, 0.0 );

   // In the sequential version, we use the Dijkstra algorithm.
   if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
      // The priority queue stores pairs of (distance, vertex)
      std::priority_queue< std::pair< Real, Index >,
                           std::vector< std::pair< Real, Index > >,
                           std::greater< std::pair< Real, Index > > >
         pq;
      pq.emplace( 0, start );

      while( ! pq.empty() ) {
         Real current_distance;
         Index current;
         std::tie( current_distance, current ) = pq.top();
         pq.pop();

         if( current_distance > distances[ current ] ) {
            continue;
         }

         const auto row = graph.getAdjacencyMatrix().getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& edge_weight = row.getValue( i );
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< Index > )
               continue;
            double distance = current_distance + edge_weight;

            if( distance < distances[ neighbor ] ) {
               distances[ neighbor ] = distance;
               pq.emplace( distance, neighbor );
            }
         }
      }
   }
   else {
      parallelSingleSourceShortestPath( graph, start, distances, launchConfig );
   }
   distances.forAllElements(
      [] __cuda_callable__( Index i, Real & x )
      {
         x = ( x == std::numeric_limits< Real >::max() ) ? -1.0 : x;
      } );
}

}  // namespace TNL::Graphs
