// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Algorithms {
namespace Graphs {


template< typename Matrix, typename Vector, typename Index = typename Matrix::IndexType >
void breadthFirstSearchTransposed( const Matrix& adjacencyMatrix, Index start, Vector& x )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( x.getSize() == adjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   using Real = typename Matrix::RealType;
   const Index n = adjacencyMatrix.getRows();

   Vector y( x.getSize() );
   x = 0;
   x.setElement( start, 1 );
   y = x;

   for( Index i = 2; i <= n; i++ )
   {
      auto x_view = x.getView();
      auto y_view = y.getView();

      auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const Real& value ) -> Real {
         return x_view[ columnIdx ] * value;
      };
      auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
         if( value && x_view[ rowIdx ] == 0 )
            y_view[ rowIdx ] = i;
      };
      adjacencyMatrix.reduceAllRows( fetch, TNL::Plus{}, keep, ( Index ) 0 );
      x = y;
   }
   x -= 1;
}

template< typename Matrix, typename Vector, typename Index = typename Matrix::IndexType >
void breadthFirstSearch( const Matrix& adjacencyMatrix, Index start, Vector& x )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );
   TNL_ASSERT_TRUE( x.getSize() == adjacencyMatrix.getRows(), "v must have the same size as the number of rows in adjacencyMatrix" );

   Matrix transposed;
   transposed.transpose( adjacencyMatrix );
   breadthFirstSearchTransposed( transposed, start, x );
}


      } // namespace Graphs
   }  // namespace Algorithms
}  // namespace TNL
