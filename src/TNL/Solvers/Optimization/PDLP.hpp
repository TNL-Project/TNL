// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
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
   int iter = 0;
   IndexType k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;

   while( iter++ < 250 ) {  //this->nextIteration() ) {
      if( adaptiveStep( G, GT, A, AT, h, b, u, l, c, x, y, k, current_omega, current_eta ) )
         return true;
      k++;
   }
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
                                             VectorType& x,
                                             VectorType& y,
                                             const IndexType k,
                                             const RealType& current_omega,
                                             RealType& current_eta )
{
   const IndexType m1 = G.getRows();
   const IndexType m2 = A.getRows();
   const IndexType n = G.getColumns();

   VectorType KT_y1( n ), KT_y2( n ), Kx( m1 + m2 );
   VectorType delta_y( m1 + m2, 0 ), delta_x( n, 0 ), aux( n, 0 );
   VectorView y1 = y.getView( 0, m1 );
   VectorView y2 = y.getView( m1, m1 + m2 );
   VectorView delta_y1 = delta_y.getView( 0, m1 );
   VectorView delta_y2 = delta_y.getView( m1, m1 + m2 );
   VectorView Kx_1 = Kx.getView( 0, m1 );
   VectorView Kx_2 = Kx.getView( m1, m1 + m2 );

   // TODO: Optimize this code - it should not be here
   const VectorType last_x( x ), last_y( y );
   const RealType max_norm = max( Matrices::maxNorm( G ), Matrices::maxNorm( A ) );

   while( true ) {
      const RealType tau = current_eta / current_omega;
      const RealType sigma = current_eta * current_omega;
      std::cout << "ITER: " << k << " TAU: " << tau << " SIGMA: " << sigma << " x: " << x << std::endl;

      GT.vectorProduct( y1, KT_y1 );
      AT.vectorProduct( y2, KT_y2 );
      KT_y1 += KT_y2;
      x = minimum( u, maximum( l, x - tau * ( c - KT_y1 ) ) );
      aux = 2.0 * x - last_x;
      G.vectorProduct( aux, Kx_1 );
      A.vectorProduct( aux, Kx_2 );
      y1 = maximum( 0, y1 + sigma * ( h - Kx_1 ) );
      y2 = y2 + sigma * ( b - Kx_2 );

      // Compute new step size eta
      delta_x = x - last_x;
      delta_y = y - last_y;
      //std::cout << "     delta_x = " << delta_x << " delta_y = " << delta_y << std::endl;
      const RealType delta_z_norm = sqrt( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      G.vectorProduct( delta_x, Kx_1 );
      A.vectorProduct( delta_x, Kx_2 );
      const RealType div = 2.0 * abs( ( Kx_1, delta_y1 ) + ( Kx_2, delta_y2 ) );
      //std::cout << "     delta_z_norm = " << delta_z_norm << " div = " << div << std::endl;
      if( abs( div ) > 1.0e-10 ) {
         const RealType max_eta = delta_z_norm / div;
         //std::cout << "     max_eta = " << max_eta << std::endl;

         const RealType new_eta = min( ( 1.0 - pow( k + 1, -0.3 ) ) * max_eta, ( 1.0 + pow( k + 1, -0.6 ) ) * current_eta );
         //std::cout << "     new_eta = " << new_eta << std::endl;
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
