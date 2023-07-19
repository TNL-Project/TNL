// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>

#include <TNL/Devices/Sequential.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Matrices/MatrixBase.h>

namespace TNL::Graphs {

template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void breadthFirstSearchTransposed_impl( const Matrix& transposedAdjacencyMatrix,
                                        typename Matrix::IndexType start,
                                        Vector& distances,
                                        Visitor&& visitor,
                                        Explorer&& explorer )
{
   TNL_ASSERT_TRUE( transposedAdjacencyMatrix.getRows() == transposedAdjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == transposedAdjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   const Index n = transposedAdjacencyMatrix.getRows();

   Vector y( distances.getSize() );
   distances = 0;
   distances.setElement( start, 1 );
   y = distances;

   for( Index i = 2; i <= n; i++ )
   {
      auto x_view = distances.getView();
      auto y_view = y.getView();

      auto fetch = [=] __cuda_callable__ ( Index rowIdx, Index columnIdx, const Real& value ) -> bool {
         return x_view[ columnIdx ] && value;
      };
      // NVCC does not allow use of if constexpr inside lambda.
      auto fetch_with_explorer = [=] __cuda_callable__ ( Index rowIdx, Index columnIdx, const Real& value ) -> bool {
         if( x_view[ columnIdx ] != 0 )
            explorer( rowIdx );
         return x_view[ columnIdx ] && value;
      };
      auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
         if( value && x_view[ rowIdx ] == 0 ) {
            y_view[ rowIdx ] = i;
            visitor( rowIdx, i );
         }
      };
      if constexpr( haveExplorer )
         transposedAdjacencyMatrix.reduceAllRows( fetch_with_explorer, TNL::Plus{}, keep, ( Index ) 0 );
      else
         transposedAdjacencyMatrix.reduceAllRows( fetch, TNL::Plus{}, keep, ( Index ) 0 );
      if( distances == y )
         break;
      distances = y;
   }
   distances -= 1;
}

template< typename Matrix, typename Vector >
void breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix,
                                   typename Matrix::IndexType start,
                                   Vector& distances )
{
   using Index = typename Matrix::IndexType;
   breadthFirstSearchTransposed_impl< false >( transposedAdjacencyMatrix,
                                               start, distances,
                                               [] __cuda_callable__ ( Index, Index ) {},
                                               [] __cuda_callable__ ( Index ) {} );
}

template< typename Matrix, typename Vector, typename Visitor >
void breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix,
                                   typename Matrix::IndexType start,
                                   Vector& distances,
                                   Visitor&& visitor )
{
   using Index = typename Matrix::IndexType;
   breadthFirstSearchTransposed_impl< false >( transposedAdjacencyMatrix,
                                               start, distances,
                                               visitor,
                                               [] __cuda_callable__ ( Index ) {} );
}

template< typename Matrix, typename Vector, typename Visitor, typename Explorer >
void breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix,
                                   typename Matrix::IndexType start,
                                   Vector& distances,
                                   Visitor&& visitor,
                                   Explorer&& explorer )
{
   breadthFirstSearchTransposed_impl< true >( transposedAdjacencyMatrix,
                                              start, distances,
                                              visitor,
                                              explorer );
}


template< bool haveExplorer, typename Matrix, typename Vector, typename Visitor, typename Explorer >
void breadthFirstSearch_impl( const Matrix& adjacencyMatrix,
                              typename Matrix::IndexType start,
                              Vector& distances,
                              Visitor&& visitor,
                              Explorer&& explorer )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == adjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   using Index = typename Matrix::IndexType;
   using Device = typename Matrix::DeviceType;

   if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value )
   {
      distances = -1;
      distances.setElement( start, 0.0 );

      std::queue< Index > q;
      q.push( start );

      while( !q.empty() ) {
         Index current = q.front();
         q.pop();

         const auto row = adjacencyMatrix.getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< Index > )
               continue;

            if constexpr( haveExplorer )
               explorer( neighbor );
            if( distances[neighbor] == -1 ) {
               Index distance = distances[ current ] + 1;
               distances[neighbor] = distance;
               visitor( neighbor, distance );
               q.push(neighbor);
            }
         }
      }
   }
   else
   {
      Matrix transposed;
      transposed.getTransposition( adjacencyMatrix );
      breadthFirstSearchTransposed( transposed, start, distances, visitor, explorer );
   }
}

template< typename Graph, typename Vector >
void breadthFirstSearch( const Graph& graph,
                         typename Graph::IndexType start,
                         Vector& distances )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >( graph.getAdjacencyMatrix(),
                                     start, distances,
                                     [] __cuda_callable__ ( Index, Index ) {},
                                     [] __cuda_callable__ ( Index ) {} );
}

template< typename Graph, typename Vector, typename Visitor >
void breadthFirstSearch( const Graph& graph,
                         typename Graph::IndexType start,
                         Vector& distances,
                         Visitor&& visitor )
{
   using Index = typename Graph::IndexType;
   breadthFirstSearch_impl< false >( graph.getAdjacencyMatrix(),
                                     start, distances,
                                     visitor,
                                     [] __cuda_callable__ ( Index ) {} );
}

template< typename Graph, typename Vector, typename Visitor, typename Explorer >
void breadthFirstSearch( const Graph& graph,
                         typename Graph::IndexType start,
                         Vector& distances,
                         Visitor&& visitor,
                         Explorer&& explorer )
{
   breadthFirstSearch_impl< true >( graph.getAdjacencyMatrix(),
                                    start, distances,
                                    visitor,
                                    explorer   );
}

}  // namespace TNL::Graphs
