// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Diagonal.h"

#include <TNL/Algorithms/parallelFor.h>

namespace TNL::Solvers::Linear::Preconditioners {

template< typename Matrix >
void
Diagonal< Matrix >::update( const MatrixPointer& matrixPointer )
{
   if( matrixPointer->getRows() == 0 )
      throw std::invalid_argument( "Diagonal::update: the matrix is empty" );
   if( matrixPointer->getRows() != matrixPointer->getColumns() )
      throw std::invalid_argument( "Diagonal::update: matrix must be square" );

   diagonal.setSize( matrixPointer->getRows() );

   VectorViewType diag_view( diagonal );

   const auto kernel_matrix = matrixPointer->getConstView();

   // TODO: Rewrite this with SparseMatrix::forAllElements
   auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      diag_view[ i ] = kernel_matrix.getElement( i, i );
   };

   Algorithms::parallelFor< DeviceType >( 0, diagonal.getSize(), kernel );
}

template< typename Matrix >
void
Diagonal< Matrix >::solve( ConstVectorViewType b, VectorViewType x ) const
{
   x = b / diagonal;
}

template< typename Matrix >
void
Diagonal< Matrices::DistributedMatrix< Matrix > >::update( const MatrixPointer& matrixPointer )
{
   diagonal.setSize( matrixPointer->getLocalMatrix().getRows() );

   LocalViewType diag_view( diagonal );
   const auto matrix_view = matrixPointer->getLocalMatrix().getConstView();

   if( matrixPointer->getRows() == matrixPointer->getColumns() ) {
      // square matrix, assume global column indices
      const auto row_range = matrixPointer->getLocalRowRange();
      auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
      {
         const IndexType gi = row_range.getGlobalIndex( i );
         diag_view[ i ] = matrix_view.getElement( i, gi );
      };
      Algorithms::parallelFor< DeviceType >( 0, diagonal.getSize(), kernel );
   }
   else {
      // non-square matrix, assume ghost indexing
      if( matrixPointer->getLocalMatrix().getRows() > matrixPointer->getLocalMatrix().getColumns() )
         throw std::invalid_argument( "Diagonal::update: the local matrix should have more columns than rows" );
      auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
      {
         diag_view[ i ] = matrix_view.getElement( i, i );
      };
      Algorithms::parallelFor< DeviceType >( 0, diagonal.getSize(), kernel );
   }
}

template< typename Matrix >
void
Diagonal< Matrices::DistributedMatrix< Matrix > >::solve( ConstVectorViewType b, VectorViewType x ) const
{
   ConstLocalViewType diag_view( diagonal );
   const auto b_view = b.getConstLocalView();
   auto x_view = x.getLocalView();

   // compute without ghosts (diagonal includes only local rows)
   x_view = b_view / diag_view;

   // synchronize ghosts
   x.startSynchronization();
}

}  // namespace TNL::Solvers::Linear::Preconditioners
