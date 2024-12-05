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

namespace TNL::Graphs {

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearchParallel( const Matrix& adjacencyMatrix,
                            typename Matrix::IndexType start,
                            Vector& distances,
                            Visitor&& visitor,
                            Explorer&& explorer )
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
   //auto distances_view = distances.getView();
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
            } );
      else
         adjacencyMatrix.forElements(
            frontier,
            0,
            frontier_size,
            /*[ = ] __cuda_callable__( Index rowIdx )
            {
               return distances_view[ rowIdx ] == i;
            },*/
            [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
            {
               TNL_ASSERT_LT( rowIdx, y_view.getSize(), "" );
               TNL_ASSERT_LT( columnIdx, y_view.getSize(), "" );
               if( columnIdx != Matrices::paddingIndex< Index > && y_view[ columnIdx ] == -1 ) {
                  atomicMax( &y_view[ columnIdx ], i + 1 );
                  atomicMax( &predecesors_view[ columnIdx ], rowIdx );
                  atomicMax( &marks_view[ columnIdx ], 1 );
               }
            } );
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

      //if( y == distances )
      //   break;
      /*std::cout << "Iteration = " << i << std::endl;
      std::cout << "Marks = " << marks << std::endl;
      std::cout << "Marks scan = " << marks_scan << std::endl;
      std::cout << "Frontier = " << frontier << std::endl;
      std::cout << "Frontier size = " << frontier_size << std::endl;*/
      distances = y;
   }
}

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearchTransposed_impl( const Matrix& transposedAdjacencyMatrix,
                                   typename Matrix::IndexType start,
                                   Vector& distances,
                                   Visitor&& visitor,
                                   Explorer&& explorer )
{
   TNL_ASSERT_TRUE( transposedAdjacencyMatrix.getRows() == transposedAdjacencyMatrix.getColumns(),
                    "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == transposedAdjacencyMatrix.getRows(),
                    "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   const Index n = transposedAdjacencyMatrix.getRows();

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
      // NVCC does not allow use of if constexpr inside lambda.
      auto fetch_with_explorer = [ = ] __cuda_callable__( Index rowIdx, Index columnIdx, const Real& value ) -> bool
      {
         if( x_view[ columnIdx ] != 0 )
            explorer( rowIdx );
         return x_view[ columnIdx ] && value;
      };
      auto keep = [ = ] __cuda_callable__( int rowIdx, const double& value ) mutable
      {
         if( value && x_view[ rowIdx ] == 0 ) {
            y_view[ rowIdx ] = i;
            visitor( rowIdx, i );
         }
      };
      if constexpr( haveExplorer )
         transposedAdjacencyMatrix.reduceAllRows( fetch_with_explorer, TNL::Plus{}, keep, (Index) 0 );
      else
         transposedAdjacencyMatrix.reduceAllRows( fetch, TNL::Plus{}, keep, (Index) 0 );
      if( distances == y )
         break;
      distances = y;
   }
   distances -= 1;
}

template< typename Matrix, typename Vector >
void
breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix, typename Matrix::IndexType start, Vector& distances )
{
   using Index = typename Matrix::IndexType;
   breadthFirstSearchTransposed_impl< false >(
      transposedAdjacencyMatrix, start, distances, [] __cuda_callable__( Index, Index ) {}, [] __cuda_callable__( Index ) {} );
}

template< typename Matrix, typename Vector, typename Visitor >
void
breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix,
                              typename Matrix::IndexType start,
                              Vector& distances,
                              Visitor&& visitor )
{
   using Index = typename Matrix::IndexType;
   breadthFirstSearchTransposed_impl< false >(
      transposedAdjacencyMatrix, start, distances, visitor, [] __cuda_callable__( Index ) {} );
}

template< typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix,
                              typename Matrix::IndexType start,
                              Vector& distances,
                              Visitor&& visitor,
                              Explorer&& explorer )
{
   breadthFirstSearchTransposed_impl< true >( transposedAdjacencyMatrix, start, distances, visitor, explorer );
}

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearch_impl( const Matrix& adjacencyMatrix,
                         typename Matrix::IndexType start,
                         Vector& distances,
                         Visitor&& visitor,
                         Explorer&& explorer )
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
      Matrix transposed;
      transposed.getTransposition( adjacencyMatrix );
      //breadthFirstSearchTransposed( transposed, start, distances, visitor, explorer );
      if constexpr( haveExplorer )
         breadthFirstSearchParallel< true >( adjacencyMatrix, start, distances, visitor, explorer );
      else
         breadthFirstSearchParallel< false >( adjacencyMatrix, start, distances, visitor, [] __cuda_callable__( Index ) {} );
   }
}

template< typename Graph, typename Vector >
void
breadthFirstSearch( const Graph& graph, typename Graph::IndexType start, Vector& distances )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >(
      graph.getAdjacencyMatrix(), start, distances, [] __cuda_callable__( Index, Index ) {}, [] __cuda_callable__( Index ) {} );
}

template< typename Graph, typename Vector, typename Visitor >
void
breadthFirstSearch( const Graph& graph, typename Graph::IndexType start, Vector& distances, Visitor&& visitor )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >( graph.getAdjacencyMatrix(), start, distances, visitor, [] __cuda_callable__( Index ) {} );
}

template< typename Graph, typename Vector, typename Visitor, typename Explorer >
void
breadthFirstSearch( const Graph& graph,
                    typename Graph::IndexType start,
                    Vector& distances,
                    Visitor&& visitor,
                    Explorer&& explorer )
{
   breadthFirstSearch_impl< true >( graph.getAdjacencyMatrix(), start, distances, visitor, explorer );
}

}  // namespace TNL::Graphs
