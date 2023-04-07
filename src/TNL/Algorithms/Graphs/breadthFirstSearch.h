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

namespace TNL {
namespace Algorithms {
namespace Graphs {

// TODO: add support for visitor in form a of lambda function
template< typename Matrix, typename Vector, typename Index = typename Matrix::IndexType >
void breadthFirstSearchTransposed( const Matrix& transposedAdjacencyMatrix, Index start, Vector& distances )
{
   TNL_ASSERT_TRUE( transposedAdjacencyMatrix.getRows() == transposedAdjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == transposedAdjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   const Index n = transposedAdjacencyMatrix.getRows();

   Vector y( distances.getSize() );
   distances = 0;
   distances.setElement( start, 1 );
   y = distances;

   for( Index i = 2; i <= n; i++ )
   {
      auto x_view = distances.getView();
      auto y_view = y.getView();

      auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const Real& value ) -> Real {
         return x_view[ columnIdx ] * value;
      };
      auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
         if( value && x_view[ rowIdx ] == 0 )
            y_view[ rowIdx ] = i;
      };
      transposedAdjacencyMatrix.reduceAllRows( fetch, TNL::Plus{}, keep, ( Index ) 0 );
      distances = y;
   }
   distances -= 1;
}

template< typename Matrix, typename Vector, typename Index = typename Matrix::IndexType >
void breadthFirstSearch( const Matrix& adjacencyMatrix, Index start, Vector& distances )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( distances.getSize() == adjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

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
            if( neighbor == adjacencyMatrix.getPaddingIndex() )
               continue;

            if( distances[neighbor] == -1 ) {
                distances[neighbor] = distances[ current ] + 1;
                q.push(neighbor);
            }
         }
      }
   }
   else
   {
      Matrix transposed;
      transposed.transpose( adjacencyMatrix );
      breadthFirstSearchTransposed( transposed, start, distances );
   }
}


      } // namespace Graphs
   }  // namespace Algorithms
}  // namespace TNL
