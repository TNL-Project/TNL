// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <assert.h>
#include <string>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <math.h>
#include <fstream>

#include <TNL/Assert.h>
#include <TNL/Containers/StaticVector.h>
#include "GEM.h"

namespace TNL::Solvers::Linear {

/*template< typename Matrix, typename Real, typename SolverMonitor >
void
GEM< Matrix, Real, SolverMonitor >::setMatrix( const MatrixPointer& matrix )
{
   TNL_ASSERT_EQ( matrix->getRows(), matrix->getColumns(), "The matrix is not square." );
   this->A = matrix;
}*/

template< typename Matrix, typename Real, typename SolverMonitor >
void
GEM< Matrix, Real, SolverMonitor >::setPivoting( bool pivoting )
{
   this->pivoting = pivoting;
}

template< typename Matrix, typename Real, typename SolverMonitor >
bool
GEM< Matrix, Real, SolverMonitor >::getPivoting() const
{
   return this->pivoting;
}

template< typename Matrix, typename Real, typename SolverMonitor >
bool
GEM< Matrix, Real, SolverMonitor >::solve( ConstVectorViewType b, VectorViewType x )
{
   MatrixType A;
   A = *this->matrix;
   if( ! solve( A, b, x ) )
      return false;
   VectorType Ax( A.getColumns() );
   this->matrix->vectorProduct( x, Ax );
   this->setResidue( l2Norm( b - Ax ) );
   return true;
}

template< typename Matrix, typename Real, typename SolverMonitor >
bool
GEM< Matrix, Real, SolverMonitor >::solve( MatrixType& A, ConstVectorViewType b, VectorViewType x )
{
   using CoordinateType = typename Containers::StaticVector< 2, IndexType >;
   TNL_ASSERT_EQ( b.getSize(), x.getSize(), "The sizes of of vectors x and b do not match." );

   if constexpr( Matrices::is_dense_matrix_v< Matrix > ) {
      const int n = A.getRows();
      auto matrix_view = A.getView();
      x = b;
      auto x_view = x.getView();
      this->success = false;
      this->setResidue( NAN );
      this->setMaxIterations( n + 1 );
      this->resetIterations();

      for( int k = 0; k < n; k++ ) {
         this->nextIteration();

         RealType pivot_value;
         IndexType pivot_position( k );

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

            // The following is to avoid compiler warnings about capturing structured bindings in C++17 later in the lambda
            // functions.
            pivot_position = pivot_position_;
            pivot_value = matrix_view.getElement( pivot_position, k );  // pivot_value_ is the maximum !!!absolute!!! value
         }
         else {
            pivot_value = matrix_view.getElement( k, k );
            pivot_position = k;
         }

         if( pivot_value == 0.0 )
            throw std::runtime_error( "Zero pivot has appeared in step " + convertToString( k ) + ". GEM has failed." );

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
                                                   else
                                                      matrix_view( k, i ) /= pivot_value;
                                                } );

         // Perform the Gauss-Jordan elimination
         Algorithms::parallelFor< DeviceType >( CoordinateType{ 0, k },
                                                CoordinateType{ n, n },
                                                [ = ] __cuda_callable__( const CoordinateType c ) mutable
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
      this->setResidue( 0 );  // The original matrix is not available anymore and so we cannot compute the true residue.
      this->success = true;
      return true;
   }
   else
      throw std::runtime_error( "GEM currently works only with dense matrices." );
}

template< typename Matrix, typename Real, typename SolverMonitor >
bool
GEM< Matrix, Real, SolverMonitor >::succeeded() const
{
   return this->success;
}

template< typename Matrix, typename Real, typename SolverMonitor >
void
GEM< Matrix, Real, SolverMonitor >::print( std::ostream& str ) const
{
   const IndexType n = this->matrix.getRows();
   const int precision( 18 );
   const std::string zero( "." );
   for( int row = 0; row < n; row++ ) {
      str << "| ";
      for( int column = 0; column < n; column++ ) {
         const RealType value = this->matrix.getElement( row, column );
         if( value == 0.0 )
            str << std::setw( precision + 6 ) << zero;
         else
            str << std::setprecision( precision ) << std::setw( precision + 6 ) << value;
      }
   }
}

}  // namespace TNL::Solvers::Linear
