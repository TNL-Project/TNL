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

namespace TNL::Graphs {

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearchParallel( const Matrix& adjacencyMatrix,
                            typename Matrix::IndexType start,
                            Vector& distances,
                            Visitor&& visitor,
                            Explorer&& explorer,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == adjacencyMatrix.getRows(),
                    "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   using Device = typename Matrix::DeviceType;
   using Index = typename Matrix::IndexType;
   const Index n = adjacencyMatrix.getRows();

   Vector y( distances.getSize() );
   Containers::Vector< Index, Device, Index > predecesors( n, -1 ), marks( n ), marks_scan( n, 0 ), frontier( n, 0 );
   distances = -1;
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
               if( columnIdx != Matrices::paddingIndex< Index > && y_view[ columnIdx ] == -1 ) {
#pragma omp atomic write
                  y_view[ columnIdx ] = i + 1;
#pragma omp atomic write
                  predecesors_view[ columnIdx ] = rowIdx;
#pragma omp atomic write
                  marks_view[ columnIdx ] = 1;
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
               if( columnIdx != Matrices::paddingIndex< Index > && y_view[ columnIdx ] == -1 ) {
                  atomicMax( &y_view[ columnIdx ], i + 1 );
                  atomicMax( &predecesors_view[ columnIdx ], rowIdx );
                  atomicMax( &marks_view[ columnIdx ], 1 );
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

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearch_impl( const Matrix& adjacencyMatrix,
                         typename Matrix::IndexType start,
                         Vector& distances,
                         Visitor&& visitor,
                         Explorer&& explorer,
                         const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == adjacencyMatrix.getRows(),
                    "v must have the same size as the number of rows in adjacencyMatrix" );

   using Index = typename Matrix::IndexType;
   using Device = typename Matrix::DeviceType;

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

            if constexpr( haveExplorer )
               explorer( neighbor );
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
      if constexpr( haveExplorer )
         breadthFirstSearchParallel< true >( adjacencyMatrix, start, distances, visitor, explorer, launchConfig );
      else
         breadthFirstSearchParallel< false >(
            adjacencyMatrix, start, distances, visitor, [] __cuda_callable__( Index ) {}, launchConfig );
   }
}

template< typename Graph, typename Vector >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >(
      graph.getAdjacencyMatrix(),
      start,
      distances,
      [] __cuda_callable__( Index, Index ) {},
      [] __cuda_callable__( Index ) {},
      launchConfig );
}

template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    Visitor&& visitor,
                    Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >(
      graph.getAdjacencyMatrix(), start, distances, visitor, [] __cuda_callable__( Index ) {}, launchConfig );
}

template< typename Graph, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    Visitor&& visitor,
                    Explorer&& explorer,
                    Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() )
{
   breadthFirstSearch_impl< true >( graph.getAdjacencyMatrix(), start, distances, visitor, explorer, launchConfig );
}

}  // namespace TNL::Graphs
