// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Containers/NDArray.h>
#include "PDLP.h"
#include "LinearTrustRegion.h"

namespace TNL::Solvers::Optimization {

template< typename LPProblem_, typename SolverMonitor >
bool
PDLP< LPProblem_, SolverMonitor >::solve( const LPProblemType& lpProblem, VectorType& x )
{
   using Array2D =
      Containers::NDArray< RealType, Containers::SizesHolder< IndexType, 0, 0 >, std::index_sequence< 0, 1 >, DeviceType >;

   const MatrixType& GA = lpProblem.getConstraintMatrix();
   const VectorType& hb = lpProblem.getConstraintVector();
   const IndexType m1 = lpProblem.getInequalityCount();
   const VectorType& l = lpProblem.getLowerBounds();
   const VectorType& u = lpProblem.getUpperBounds();
   const VectorType& c = lpProblem.getObjectiveFunction();

   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();
   const IndexType N = n + m;
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

   if( restarting == PDLPRestarting::KKTError )
      this->primal_gradient.setSize( n );
   IndexType k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;

   const IndexType max_restarting_steps = 100;
   Array2D z_container;
   z_container.setSizes( max_restarting_steps + 1, n + m1 + m2 );
   auto z_container_view = z_container.getView();
   VectorType z_c( N ), z_bar( N ), last_z( N ), eta_container( max_restarting_steps + 1 );
   auto z_bar_view = z_bar.getView();
   z_bar_view = 0;
   auto z_c_view = z_c.getView();
   z_c = 0;
   last_z = 0;
   auto eta_container_view = eta_container.getView();
   RealType eta_sum( 0 ), last_z_gap( 0 );

   while( k < 10000 ) {  //this->nextIteration() ) {
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
         VectorView z_new_view( &z_container( t + 1, 0 ), n + m1 + m2 );
         VectorView out_x_view = z_new_view.getView( 0, n );
         VectorView out_y_view = z_new_view.getView( n, n + m1 + m2 );
         if( adaptiveStep(
                GA, GAT, hb, m1, u, l, c, in_x_view, in_y_view, out_x_view, out_y_view, k, current_omega, current_eta ) )
            return true;
         eta_container[ t + 1 ] = current_eta;
         eta_sum += current_eta;
         Algorithms::parallelFor< DeviceType >( 0,
                                                n + m1 + m2,
                                                [ = ] __cuda_callable__( IndexType i ) mutable
                                                {
                                                   z_bar_view[ i ] = 0;
                                                   for( IndexType j = 0; j <= t + 1; j++ )
                                                      z_bar_view[ i ] +=
                                                         z_container_view( j, i ) * eta_container_view[ j ] / eta_sum;
                                                } );
         t++;
         k++;

         RealType z_new_gap, z_bar_gap, error;
         if( restarting == PDLPRestarting::KKTError )
            error = KKTError( GA, GAT, m1, hb, out_x_view, out_y_view, u, l, c, current_omega );
         else if( restarting == PDLPRestarting::DualityGap ) {
            // Solve argmin_{x^hat \in X, y^hat \in Y } [ ((K^T *y )^T - c )*x^hat + ( q - K*x )^T *y^hat ]
            z_new_gap = primalDualGap( GA, GAT, m1, c, hb, u, l, z_new_view, z_view );
            z_bar_gap = primalDualGap( GA, GAT, m1, c, hb, u, l, z_bar_view, z_view );
            // Get restart candidate
            error = min( z_new_gap, z_bar_gap );
            z_c = z_new_gap < z_bar_gap ? z_new_view : z_bar_view;
            if( abs( min( z_new_gap, z_bar_gap ) ) < 1.0e-5 ) {
               std::cout << "Found solution with duality gap: " << min( z_new_gap, z_bar_gap ) << std::endl;
               std::cout << "x: " << z_c.getView( 0, n ) << std::endl;
               std::cout << "y: " << z_c.getView( n, N ) << std::endl;
               return true;
            }
            std::cout << "ITER: " << k << " / " << t << " COST: " << dot( z_c.getView( 0, n ), x ) << " ERROR:" << error;

            // Restart criteria
            const RealType beta_sufficient = 0.9;
            const RealType beta_necessary = 0.1;
            const RealType beta_artificial = 0.5;
            last_z_gap = primalDualGap( GA, GAT, m1, c, hb, u, l, z_view, last_z );
            VectorView z_n_t_view( &z_container( t, 0 ), N );
            const RealType z_n_t_gap = primalDualGap( GA, GAT, m1, c, hb, u, l, z_n_t_view, z_view );

            if( z_new_gap <= beta_sufficient * last_z_gap
                || ( z_new_gap <= beta_necessary * last_z_gap && z_new_gap > z_n_t_gap ) || ( t >= beta_artificial * k ) )
            {
               std::cout << " ... RESTARTING \n";
               break;
            }
         }
         std::cout << std::endl;
      }
      last_z = z_view;
      auto new_x_view = z_c.getView( 0, n );
      auto new_y_view = z_c.getView( n, n + m1 + m2 );
      //Compute new parameter omega
      RealType delta_x = lpNorm( new_x_view - x, 2 );
      RealType delta_y = lpNorm( new_y_view - y, 2 );
      if( delta_x > 1.0e-10 && delta_y > 1.0e-10 ) {
         const RealType theta = 0.5;
         current_omega = exp( theta * log( delta_y / delta_x ) + ( 1.0 - theta ) * log( current_omega ) );
         //std::cout << "Setting new omega: " << current_omega << std::endl;
      }
      x = new_x_view;
      y = new_y_view;
   }
   return false;
}

template< typename LPProblem_, typename SolverMonitor >
bool
PDLP< LPProblem_, SolverMonitor >::adaptiveStep( const MatrixType& GA,
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
                                                 RealType& current_omega,
                                                 RealType& current_eta )
{
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
      if( restarting == PDLPRestarting::KKTError ) {
         primal_gradient = c - KT_y;
         out_x = minimum( u, maximum( l, in_x - tau * primal_gradient ) );
      }
      else
         out_x = minimum( u, maximum( l, in_x - tau * ( c - KT_y ) ) );
      aux = 2.0 * out_x - in_x;
      GA.vectorProduct( aux, Kx );
      out_y1 = maximum( 0, in_y1 + sigma * ( h - Kx_1 ) );
      if( m2 > 0 )
         out_y2 = in_y2 + sigma * ( b - Kx_2 );
      //std::cout << "    ITER: " << t << " ETA: " << current_eta << " TAU: " << tau << " SIGMA: " << sigma
      //          << " COST: " << dot( c, out_x ) << " x: " << out_x << " y: " << out_y << std::endl;

      // Compute new parameter eta
      delta_x = out_x - in_x;
      delta_y = out_y - in_y;
      const RealType delta_z_norm = sqrt( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      GA.vectorProduct( delta_x, Kx );
      const RealType div = 2.0 * abs( ( Kx, delta_y ) );
      if( abs( div ) > 1.0e-10 ) {
         const RealType max_eta = delta_z_norm / div;

         const RealType new_eta = min( ( 1.0 - pow( k + 1, -0.3 ) ) * max_eta, ( 1.0 + pow( k + 1, -0.6 ) ) * current_eta );
         //std::cout << "NEW ETA: " << new_eta << " MAX ETA: " << max_eta << std::endl;
         if( new_eta < max_eta ) {
            current_eta = new_eta;
            return false;
         }
         else {
            current_eta = new_eta;
         }
      }
      else {
         current_eta = 1.0 / max_norm;
         return false;
      }
   }
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::primalDualGap( const MatrixType& GA,
                                                  const MatrixType& GAT,
                                                  const IndexType m1,
                                                  const VectorType& c,
                                                  const VectorType& q,
                                                  const VectorType& u,
                                                  const VectorType& l,
                                                  const VectorView& z,
                                                  const VectorView& z_ref ) const -> RealType
{
   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();
   const IndexType N = n + m;

   auto x_view = z.getConstView( 0, n );
   auto y_view = z.getConstView( n, N );

   VectorType g( N ), g_l( N ), g_u( N );
   auto g_1 = g.getView( 0, n );
   auto g_2 = g.getView( n, N );
   GAT.vectorProduct( y_view, g_1 );
   g_1 = c - g_1;
   GA.vectorProduct( x_view, g_2 );
   g_2 = g_2 - q;

   g_l.getView( 0, n ) = l;
   g_u.getView( 0, n ) = u;
   if( m1 > 0 )
      g_l.getView( n, n + m1 ) = 0.0;
   if( m2 > 0 )
      g_l.getView( n + m1, N ) = -std::numeric_limits< RealType >::infinity();
   g_u.getView( n, N ) = std::numeric_limits< RealType >::infinity();

   VectorType z_hat( N );
   auto z_hat_view = z_hat.getView();
   RealType r = max( 0.001, l2Norm( z_ref - z ) );  // TODO: How to deal with small r?
   //std::cout << "   z = " << z_view << " g = " << g << " g_l = " << g_l << " g_u = " << g_u << " r = " << r
   //          << std::endl;
   linearTrustRegion( z, g_l.getView(), g_u.getView(), g.getView(), r, z_hat_view );
   return ( ( c, x_view ) - ( y_view, q ) - ( z_hat, g ) ) / r;  // TODO: How to deal with small r?
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::KKTError( const MatrixType& GA,
                                             const MatrixType& GAT,
                                             const IndexType m1,
                                             const VectorType& q,
                                             const VectorView& x,
                                             const VectorView& y,
                                             const VectorType& u,
                                             const VectorType& l,
                                             const VectorType& c,
                                             const RealType& omega ) const -> RealType
{
   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();

   auto h = q.getConstView( 0, m1 );
   auto b = q.getConstView( m1, m );
   VectorType aux1( m1 ), aux2( m2 ), KTy( n ), lambda( n );
   GA.vectorProduct( x, aux1, 1, 0, 0, m1 );  // aux1 = G * x
   GA.vectorProduct( x, aux2, 1, 0, m1, m );  // aux2 = A * x
   aux1 = maximum( h - aux1, 0 );
   aux2 -= b;
   GAT.vectorProduct( y, KTy );
   auto pg = this->primal_gradient.getConstView();
   auto l_view = l.getConstView();
   auto u_view = u.getConstView();
   lambda.forAllElements(
      [ = ] __cuda_callable__( IndexType i, RealType & value )
      {
         value = max( pg[ i ], 0 ) * ( l_view[ i ] != -std::numeric_limits< RealType >::infinity() )
               + min( pg[ i ], 0 ) * ( u_view[ i ] != std::numeric_limits< RealType >::infinity() );
      } );
   /*const auto c_view = c.getConstView();
   const auto KTy_view = KTy.getConstView();
   lambda.forAllElements(
      [ = ] __cuda_callable__( IndexType i, RealType & value )
      {
         value = c_view[ i ] - KTy_view[ i ];
         if( l_view[ i ] != -std::numeric_limits< RealType >::infinity() ) {
            if( u_view[ i ] != std::numeric_limits< RealType >::infinity() )
               value = 0;
            else
               value = min( value, 0 );
         }
         else if( u_view[ i ] != std::numeric_limits< RealType >::infinity() )
            value = max( value, 0 );
      } );*/
   std::cout << "LAMBDA: " << lambda << std::endl;
   std::cout << "     l: " << l << std::endl;
   std::cout << "     u: " << u << std::endl;
   const RealType omega_sqr = omega * omega;
   const auto primal_objective = ( c, x );
   const auto primal_constraint_residue = ( aux1, aux1 ) + ( aux2, aux2 );
   const auto lambda_view = lambda.getConstView();
   const auto dual_objective =
      ( q, y )
      + Algorithms::reduce< DeviceType >( (IndexType) 0,
                                          n,
                                          [ = ] __cuda_callable__( IndexType i ) -> RealType
                                          {
                                             RealType result = 0;
                                             if( l_view[ i ] != -std::numeric_limits< RealType >::infinity() )
                                                result += l_view[ i ] * lambda_view[ i ];
                                             if( u_view[ i ] != std::numeric_limits< RealType >::infinity() )
                                                result -= u_view[ i ] * lambda_view[ i ];
                                             return result;
                                          },
                                          TNL::Plus{} );
   const RealType dual_constraint_residue = l2Norm( c - KTy - lambda );
   const RealType objective_diff = dual_objective - primal_objective;
   const RealType error = omega_sqr * primal_constraint_residue
                        + 1.0 / omega_sqr * dual_constraint_residue * dual_constraint_residue + objective_diff * objective_diff;
   std::cout << "PRIMAL OBJ.: " << primal_objective << " DUAL OBJ.: " << dual_objective << " PRIMAL CONSTR. RES."
             << primal_constraint_residue << " DUAL CONSTR. RES. " << dual_constraint_residue << " ERROR: " << sqrt( error )
             << std::endl;
   return sqrt( error );
}

}  // namespace TNL::Solvers::Optimization
