// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <TNL/Containers/StaticVector.h>
#include "GEM.h"

namespace TNL::Solvers::Linear {

template< typename Matrix >
void
GEM< Matrix >::setPivoting( bool pivoting )
{
   this->pivoting = pivoting;
}

template< typename Matrix >
bool
GEM< Matrix >::getPivoting() const
{
   return this->pivoting;
}

template< typename Matrix >
bool
GEM< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   MatrixType A;
   A = *this->matrix;
   return solve( A, b, x );
}

template< typename Matrix >
bool
GEM< Matrix >::solve( MatrixType& A, ConstVectorViewType b, VectorViewType x )
{
   if( this->matrix->getColumns() != x.getSize() )
      throw std::invalid_argument( "GEM::solve: wrong size of the solution vector" );
   if( this->matrix->getColumns() != b.getSize() )
      throw std::invalid_argument( "GEM::solve: wrong size of the right hand side" );

   using CoordinateType = typename Containers::StaticVector< 2, IndexType >;

   const int n = A.getRows();
   auto matrix_view = A.getView();
   x = b;
   auto x_view = x.getView();
   this->setResidue( NAN );
   this->setMaxIterations( n + 1 );
   this->resetIterations();

   for( int k = 0; k < n; k++ ) {
      this->nextIteration();

      RealType pivot_value;
      IndexType pivot_position = k;

      if( this->pivoting ) {
         // Find the pivot - the largest in k-th row
         auto [ pivot_value_, pivot_position_ ] = Algorithms::reduceWithArgument< DeviceType >(
            k,
            n,
            [ = ] __cuda_callable__( const IndexType rowIdx ) -> RealType
            {
               return abs( matrix_view( rowIdx, k ) );
            },
            TNL::MaxWithArg{} );

         pivot_position = pivot_position_;
         // ignore pivot_value_ - it is the maximum **absolute** value
         pivot_value = matrix_view.getElement( pivot_position, k );
      }
      else {
         pivot_value = matrix_view.getElement( k, k );
         pivot_position = k;
      }

      if( pivot_value == 0.0 )
         throw std::runtime_error( "Zero pivot has appeared in step " + std::to_string( k ) + ". GEM has failed." );

      // Swap the rows ...
      if( pivot_position != k ) {
         Algorithms::parallelFor< DeviceType >( k,
                                                n,
                                                [ = ] __cuda_callable__( const IndexType i ) mutable
                                                {
                                                   swap( matrix_view( k, i ), matrix_view( pivot_position, i ) );
                                                   if( i == k ) {
                                                      swap( x_view[ k ], x_view[ pivot_position ] );
                                                   }
                                                } );
      }

      // Divide the k-th row by pivot (including the b vector)
      Algorithms::parallelFor< DeviceType >( k,
                                             n,
                                             [ = ] __cuda_callable__( const IndexType i ) mutable
                                             {
                                                if( i == k ) {
                                                   matrix_view( k, i ) = 1.0;
                                                   x_view[ k ] /= pivot_value;
                                                }
                                                else {
                                                   matrix_view( k, i ) /= pivot_value;
                                                }
                                             } );

      // Perform the Gauss-Jordan elimination
      Algorithms::parallelFor< DeviceType >( CoordinateType{ 0, k },
                                             CoordinateType{ n, n },
                                             [ = ] __cuda_callable__( const CoordinateType& c ) mutable
                                             {
                                                const auto& i = c[ 0 ];
                                                const auto& j = c[ 1 ];
                                                if( i != k ) {
                                                   // Subtract the k-th row from the current row
                                                   if( j > k )
                                                      matrix_view( i, j ) -= matrix_view( i, k ) * matrix_view( k, j );
                                                   else
                                                      x_view[ i ] -= matrix_view( i, k ) * x_view[ k ];
                                                }
                                             } );

      // Set the k-th column to zero for all rows except the k-th row
      Algorithms::parallelFor< DeviceType >( 0,
                                             n,
                                             [ = ] __cuda_callable__( const IndexType i ) mutable
                                             {
                                                if( i != k )
                                                   matrix_view( i, k ) = 0.0;
                                             } );
   }
   // Direct solvers set the residue to zero.
   // (And the original matrix is not available anymore, so we cannot compute it anyway.)
   this->setResidue( 0 );
   return true;
}

}  // namespace TNL::Solvers::Linear
