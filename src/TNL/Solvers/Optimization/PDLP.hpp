// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Containers/NDArray.h>
#include "PDLP.h"

namespace TNL::Solvers::Optimization {

template< typename Vector, typename SolverMonitor >
template< typename AMatrixType, typename GMatrixType >
bool
PDLP< Vector, SolverMonitor >::solve( const Vector& c,
                                      const GMatrixType& G,
                                      const Vector& h,
                                      const AMatrixType& A,
                                      const Vector& b,
                                      const Vector& l,
                                      const Vector& u,
                                      Vector& x )
{
   using Array2D =
      Containers::NDArray< RealType, Containers::SizesHolder< IndexType, 0, 0 >, std::index_sequence< 0, 1 >, DeviceType >;
   const IndexType m1 = G.getRows();
   const IndexType m2 = A.getRows();
   const IndexType n = G.getColumns();
   TNL_ASSERT_EQ( c.getSize(), n, "" );
   TNL_ASSERT_EQ( h.getSize(), m1, "" );
   TNL_ASSERT_EQ( b.getSize(), m2, "" );
   TNL_ASSERT_EQ( l.getSize(), n, "" );
   TNL_ASSERT_EQ( u.getSize(), n, "" );
   TNL_ASSERT_EQ( x.getSize(), n, "" );
   AMatrixType AT;
   GMatrixType GT;
   AT.getTransposition( A );
   GT.getTransposition( G );

   VectorType y( m1 + m2, 0 );
   const RealType max_norm = max( Matrices::maxNorm( G ), Matrices::maxNorm( A ) );
   const RealType initial_eta = 1.0 / max_norm;
   const RealType c_norm = lpNorm( c, 2 );
   const RealType q_norm = sqrt( ( b, b ) + ( h, h ) );
   const RealType initial_omega = ( c_norm > 1.0e-10 && q_norm > 1.0e-10 ) ? c_norm / q_norm : 1;
   std::cout << "initital eta = " << initial_eta << " omega = " << initial_omega << std::endl;

   IndexType k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;

   const IndexType max_restarting_steps = 15;
   Array2D z_container;
   z_container.setSizes( max_restarting_steps + 1, n + m1 + m2 );
   auto z_container_view = z_container.getView();
   VectorType z_c( n + m1 + m2 ), eta_container( max_restarting_steps + 1 );
   auto z_c_view = z_c.getView();
   z_c_view = 0;
   auto eta_container_view = eta_container.getView();
   RealType eta_sum( 0 );

   while( k < 120 ) {  //this->nextIteration() ) {
      IndexType t = 0;
      eta_sum = 0;
      VectorView z_view( &z_container( 0, 0 ), n + m1 + m2 );
      z_view = z_c;
      t = 0;
      eta_container[ 0 ] = eta_sum = current_eta;
      while( t < max_restarting_steps ) {
         VectorView in_z_view( &z_container( t, 0 ), n + m1 + m2 );
         VectorView in_x_view = in_z_view.getView( 0, n );
         VectorView in_y_view = in_z_view.getView( n, n + m1 + m2 );
         VectorView out_z_view( &z_container( t + 1, 0 ), n + m1 + m2 );
         VectorView out_x_view = out_z_view.getView( 0, n );
         VectorView out_y_view = out_z_view.getView( n, n + m1 + m2 );
         if( adaptiveStep(
                G, GT, A, AT, h, b, u, l, c, in_x_view, in_y_view, out_x_view, out_y_view, k, current_omega, current_eta ) )
            return true;
         eta_container[ t + 1 ] = current_eta;
         eta_sum += current_eta;
         Algorithms::parallelFor< DeviceType >( 0,
                                                n + m1 + m2,
                                                [ = ] __cuda_callable__ ( IndexType i ) mutable
                                                {
                                                   //z_c_view[ i ] = z_container_view( t + 1, i );
                                                   z_c_view[ i ] = 0;
                                                   for( IndexType j = 0; j <= t + 1; j++ )
                                                      z_c_view[ i ] +=
                                                         z_container_view( j, i ) * eta_container_view[ j ] / eta_sum;
                                                } );
         t++;
         k++;
      }
   }
   auto z_c_x_view = z_c.getView( 0, n );
   x = z_c_x_view;
   return false;
}

template< typename Vector, typename SolverMonitor >
template< typename AMatrixType, typename GMatrixType >
bool
PDLP< Vector, SolverMonitor >::adaptiveStep( const GMatrixType& G,
                                             const GMatrixType& GT,
                                             const AMatrixType& A,
                                             const AMatrixType& AT,
                                             const VectorType& h,
                                             const VectorType& b,
                                             const VectorType& u,
                                             const VectorType& l,
                                             const VectorType& c,
                                             const VectorView& in_x,
                                             const VectorView& in_y,
                                             VectorView& out_x,
                                             VectorView& out_y,
                                             const IndexType k,
                                             const RealType& current_omega,
                                             RealType& current_eta )
{
   const IndexType m1 = G.getRows();
   const IndexType m2 = A.getRows();
   const IndexType n = G.getColumns();

   VectorType KT_y1( n ), KT_y2( n ), Kx( m1 + m2 );
   VectorType delta_y( m1 + m2, 0 ), delta_x( n, 0 ), aux( n, 0 );
   auto in_y1 = in_y.getConstView( 0, m1 );
   auto in_y2 = in_y.getConstView( m1, m1 + m2 );
   auto out_y1 = out_y.getView( 0, m1 );
   auto out_y2 = out_y.getView( m1, m1 + m2 );
   VectorView delta_y1 = delta_y.getView( 0, m1 );
   VectorView delta_y2 = delta_y.getView( m1, m1 + m2 );
   VectorView Kx_1 = Kx.getView( 0, m1 );
   VectorView Kx_2 = Kx.getView( m1, m1 + m2 );

   // TODO: Optimize this code - it should not be here
   VectorType last_x, last_y;
   last_x = in_x;
   last_y = in_y;
   const RealType max_norm = max( Matrices::maxNorm( G ), Matrices::maxNorm( A ) );

   while( true ) {
      const RealType tau = current_eta / current_omega;
      const RealType sigma = current_eta * current_omega;

      GT.vectorProduct( in_y1, KT_y1 );
      AT.vectorProduct( in_y2, KT_y2 );
      KT_y1 += KT_y2;
      out_x = minimum( u, maximum( l, in_x - tau * ( c - KT_y1 ) ) );
      aux = 2.0 * out_x - last_x;
      G.vectorProduct( aux, Kx_1 );
      A.vectorProduct( aux, Kx_2 );
      out_y1 = maximum( 0, in_y1 + sigma * ( h - Kx_1 ) );
      out_y2 = in_y2 + sigma * ( b - Kx_2 );
      std::cout << "ITER: " << k << " TAU: " << tau << " SIGMA: " << sigma << " x: " << out_x << std::endl;

      // Compute new step size eta
      delta_x = out_x - last_x;
      delta_y = out_y - last_y;
      const RealType delta_z_norm = sqrt( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      G.vectorProduct( delta_x, Kx_1 );
      A.vectorProduct( delta_x, Kx_2 );
      const RealType div = 2.0 * abs( ( Kx_1, delta_y1 ) + ( Kx_2, delta_y2 ) );
      if( abs( div ) > 1.0e-10 ) {
         const RealType max_eta = delta_z_norm / div;

         const RealType new_eta = min( ( 1.0 - pow( k + 1, -0.3 ) ) * max_eta, ( 1.0 + pow( k + 1, -0.6 ) ) * current_eta );
         if( current_eta < max_eta ) {
            current_eta = new_eta;
            return false;
         }
         else {
            current_eta = new_eta;
         }
      }
      else {
         current_eta = 1.0 / max_norm;
      }
      return false;
   }
}

template< typename Vector, typename SolverMonitor >
template< typename GMatrixType >
bool
PDLP< Vector, SolverMonitor >::solve( const Vector& c,
                                      const GMatrixType& G,
                                      const Vector& h,
                                      const Vector& l,
                                      const Vector& u,
                                      Vector& x )
{
   const IndexType m = G.getRows();
   const IndexType n = G.getColumns();
   TNL_ASSERT_EQ( c.getSize(), n, "" );
   TNL_ASSERT_EQ( h.getSize(), m, "" );
   TNL_ASSERT_EQ( l.getSize(), n, "" );
   TNL_ASSERT_EQ( u.getSize(), n, "" );
   TNL_ASSERT_EQ( x.getSize(), n, "" );
   GMatrixType GT;
   GT.getTransposition( G );

   const RealType max_norm = Matrices::maxNorm( G );
   const RealType eta = 1.0 / max_norm;
   const RealType c_norm = lpNorm( c, 2 );
   const RealType q_norm = lpNorm( h, 2 );
   RealType omega;
   if( c_norm > 1.0e-10 && q_norm > 1.0e-10 )
      omega = c_norm / q_norm;
   else
      omega = 1;
   const RealType tau = eta / omega;
   const RealType sigma = eta * omega;

   std::cout << "m = " << m << ", n = " << n << std::endl;

   VectorType y( m, 0 ), last_x( n, 0 ), GT_y( n ), Gx( m );
   int iter = 0;
   while( true && iter++ < 2500 ) {  //this->nextIteration() ) {
      last_x = x;
      GT.vectorProduct( y, GT_y );
      x -= tau * ( c - GT_y );
      x = minimum( u, maximum( l, x ) );
      last_x = 2 * x - last_x;
      G.vectorProduct( last_x, Gx );
      y += sigma * ( h - Gx );
      y = maximum( 0, y );
      std::cout << this->getIterations() << " >>> x = " << x << std::endl;
   }
   return true;
}

}  // namespace TNL::Solvers::Optimization
