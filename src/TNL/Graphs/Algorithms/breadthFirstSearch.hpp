// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "breadthFirstSearch.h"

namespace TNL::Graphs::Algorithms {

template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearchParallel( const Graph& graph,
                            typename Graph::IndexType start,
                            Vector& distances,
                            Visitor&& visitor,
                            const TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;
   using Index = typename Graph::IndexType;
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   const Index n = graph.getVertexCount();
   distances.setSize( n );

   Vector y( distances.getSize() );
   Containers::Vector< Index, Device, Index > predecessors( n, -1 );
   Containers::Vector< Index, Device, Index > marks( n );
   Containers::Vector< Index, Device, Index > marks_scan( n, 0 );
   Containers::Vector< Index, Device, Index > frontier( n, 0 );
   distances = -1;
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   Index frontier_size( 1 );
   y = distances;
   auto y_view = y.getView();
   auto predecessors_view = predecessors.getView();
   auto marks_view = marks.getView();
   auto marks_scan_view = marks_scan.getView();
   for( Index i = 0; i <= n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< Device, Devices::Host > )
         adjacencyMatrix.forElements(
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index sourceIdx, Index localIdx, Index targetIdx, const Real& weight ) mutable
            {
               if( targetIdx != Matrices::paddingIndex< Index > && y_view[ targetIdx ] == -1 ) {
#if defined( _OPENMP )
            #pragma omp atomic write
#endif
                  y_view[ targetIdx ] = i + 1;
#if defined( _OPENMP )
            #pragma omp atomic write
#endif
                  predecessors_view[ targetIdx ] = sourceIdx;
#if defined( _OPENMP )
            #pragma omp atomic write
#endif
                  marks_view[ targetIdx ] = 1;
                  visitor( targetIdx, i + 1 );
               }
            },
            launchConfig );
      else
         adjacencyMatrix.forElements(
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index sourceIdx, Index localIdx, Index targetIdx, const Real& weight ) mutable
            {
               TNL_ASSERT_GE( sourceIdx, 0, "" );
               TNL_ASSERT_LT( sourceIdx, y_view.getSize(), "" );
               TNL_ASSERT_GE( targetIdx, 0, "" );
               TNL_ASSERT_LT( targetIdx, y_view.getSize(), "" );
               if( targetIdx != Matrices::paddingIndex< Index > && y_view[ targetIdx ] == -1 ) {
                  atomicMax( &y_view[ targetIdx ], i + 1 );
                  atomicMax( &predecessors_view[ targetIdx ], sourceIdx );
                  atomicMax( &marks_view[ targetIdx ], 1 );
                  visitor( targetIdx, i + 1 );
               }
            },
            launchConfig );
      TNL::Algorithms::inclusiveScan( marks, marks_scan );
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

template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearch_impl( const Graph& graph,
                         typename Graph::IndexType start,
                         Vector& distances,
                         Visitor&& visitor,
                         const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using Index = typename Graph::IndexType;
   using Device = typename Graph::DeviceType;
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   const Index n = graph.getVertexCount();
   distances.setSize( n );

   if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
      distances = -1;
      distances.setElement( start, 0.0 );

      std::queue< Index > q;
      q.push( start );

      while( ! q.empty() ) {
         Index current = q.front();
         q.pop();

         const auto row = adjacencyMatrix.getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< Index > )
               continue;

            if( distances[ neighbor ] == -1 ) {
               Index distance = distances[ current ] + 1;
               distances[ neighbor ] = distance;
               visitor( neighbor, distance );
               q.push( neighbor );
            }
         }
      }
   }
   else {
      breadthFirstSearchParallel( graph, start, distances, visitor, launchConfig );
   }
}

template< typename Graph, typename Vector >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl( graph, start, distances, [] __cuda_callable__( Index, Index ) {}, launchConfig );
}

template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    Visitor&& visitor,
                    TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   breadthFirstSearch_impl( graph, start, distances, visitor, launchConfig );
}

}  //namespace TNL::Graphs::Algorithms
