// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Containers/NDArray.h>
#include "PDLP.h"

namespace TNL::Solvers::Optimization {

template< typename Vector, typename SolverMonitor >
template< typename MatrixType >
bool
PDLP< Vector, SolverMonitor >::solve( const Vector& c,
                                      const MatrixType& GA,
                                      const Vector& hb,
                                      const IndexType m1,
                                      const Vector& l,
                                      const Vector& u,
                                      Vector& x )
{
   using Array2D =
      Containers::NDArray< RealType, Containers::SizesHolder< IndexType, 0, 0 >, std::index_sequence< 0, 1 >, DeviceType >;
   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();
   TNL_ASSERT_EQ( c.getSize(), n, "" );
   TNL_ASSERT_EQ( hb.getSize(), m, "" );
   TNL_ASSERT_EQ( l.getSize(), n, "" );
   TNL_ASSERT_EQ( u.getSize(), n, "" );
   TNL_ASSERT_EQ( x.getSize(), n, "" );
   MatrixType GAT;
   GAT.getTransposition( GA );

   VectorType y( m1 + m2, 0 );
   const RealType max_norm = Matrices::maxNorm( GA );
   const RealType initial_eta = 1.0 / max_norm;
   const RealType c_norm = lpNorm( c, 2 );
   const RealType q_norm = l2Norm( hb );
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
                GA, GAT, hb, m1, u, l, c, in_x_view, in_y_view, out_x_view, out_y_view, k, current_omega, current_eta ) )
            return true;
         eta_container[ t + 1 ] = current_eta;
         eta_sum += current_eta;
         Algorithms::parallelFor< DeviceType >( 0,
                                                n + m1 + m2,
                                                [ = ] __cuda_callable__( IndexType i ) mutable
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
template< typename MatrixType >
bool
PDLP< Vector, SolverMonitor >::adaptiveStep( const MatrixType& GA,
                                             const MatrixType& GAT,
                                             const VectorType& hb,
                                             const IndexType m1,
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
   TNL_ASSERT_GT( m1, 0, "Number of inequalities must be positive." );
   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();

   VectorType KT_y( n ), Kx( m );
   VectorType delta_y( m, 0 ), delta_x( n, 0 ), aux( n, 0 );

   const auto h = hb.getConstView( 0, m1 );
   const auto in_y1 = in_y.getConstView( 0, m1 );
   const auto Kx_1 = Kx.getConstView( 0, m1 );
   auto out_y1 = out_y.getView( 0, m1 );

   ConstVectorView b, in_y2, Kx_2;
   VectorView out_y2;
   if( m2 > 0 ) {
      b.bind( hb.getConstView( m1, m ) );
      in_y2.bind( in_y.getConstView( m1, m ) );
      Kx_2.bind( Kx.getConstView( m1, m ) );
      out_y2.bind( out_y.getView( m1, m ) );
   }

   const RealType max_norm = Matrices::maxNorm( GA );

   while( true ) {
      const RealType tau = current_eta / current_omega;
      const RealType sigma = current_eta * current_omega;

      GAT.vectorProduct( in_y, KT_y );
      out_x = minimum( u, maximum( l, in_x - tau * ( c - KT_y ) ) );
      aux = 2.0 * out_x - in_x;
      GA.vectorProduct( aux, Kx );
      out_y1 = maximum( 0, in_y1 + sigma * ( h - Kx_1 ) );
      if( m2 > 0 )
         out_y2 = in_y2 + sigma * ( b - Kx_2 );
      std::cout << "ITER: " << k << " TAU: " << tau << " SIGMA: " << sigma << " x: " << out_x << std::endl;

      // Compute new step size eta
      delta_x = out_x - in_x;
      delta_y = out_y - in_y;
      const RealType delta_z_norm = sqrt( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      GA.vectorProduct( delta_x, Kx );
      const RealType div = 2.0 * abs( ( Kx, delta_y ) );
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

}  // namespace TNL::Solvers::Optimization
