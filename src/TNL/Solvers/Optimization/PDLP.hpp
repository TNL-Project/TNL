// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include "PDLP.h"
#include "LinearTrustRegion.h"
#include "Preconditioning/PockChambolle.h"

#include <iomanip>

namespace TNL::Solvers::Optimization {

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::solve( const LPProblemType& lpProblem, VectorType& x )
   -> std::tuple< bool, RealType, RealType >
{
   this->GA = lpProblem.getConstraintMatrix();
   this->c = lpProblem.getObjectiveFunction();
   this->hb = lpProblem.getConstraintVector();
   this->l = lpProblem.getLowerBounds();
   this->u = lpProblem.getUpperBounds();
   const IndexType m1 = lpProblem.getInequalityCount();

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

   // Preconditioning
   D1.setSize( m );
   D2.setSize( n );
   D1 = 1;
   D2 = 1;
   //Preconditioning::pockChambole( GA, GAT, m1, c, hb, l, u, D1, D2 );
   //std::cout << "D1: " << D1 << std::endl;
   //std::cout << "D2: " << D2 << std::endl;

   VectorType y( m1 + m2, 0 );
   const RealType max_norm = Matrices::maxNorm( GA );
   std::cout << "Constraint matrix max norm: " << max_norm << std::endl;
   //std::cout << "Constraint matrix: " << GA << std::endl;
   if( max_norm < 1.0e-10 )
      throw std::runtime_error( "Matrix for the LP problem is nearly zero matrix." );
   const RealType initial_eta = 1.0 / max_norm;
   const RealType c_norm = lpNorm( c, 2 );
   const RealType q_norm = l2Norm( hb );
   const RealType initial_omega = ( c_norm > 1.0e-10 && q_norm > 1.0e-10 ) ? c_norm / q_norm : 1;

   IndexType k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;

   const IndexType max_restarting_steps = 10000;
   VectorType z_c( N ), z_bar( N ), last_z( N );
   auto l_view = l.getConstView();
   auto u_view = u.getConstView();
   z_c.forElements( 0,
                    n,
                    [ = ] __cuda_callable__( IndexType i, RealType & value ) mutable
                    {
                       if( l_view[ i ] == -std::numeric_limits< RealType >::infinity() )
                          value = min( u_view[ i ], 0 );
                       else
                          value = l_view[ i ];
                    } );
   z_c.getView( n, N ) = 0;
   //z_c = 0;  // TODO: Erase
   last_z = z_c;
   x = z_c.getView( 0, n );
   y = z_c.getView( n, N );
   RealType eta_sum( 0 ), error_n_0( 0 );

   VectorType z_k_0( N ), z_k_t( N ), z_k_t_new( N );
   while( k < 1000000 ) {  //this->nextIteration() ) {
      IndexType t = 0;

      eta_sum = current_eta;
      z_k_t = z_k_0 = z_bar = z_c;
      while( t < max_restarting_steps ) {
         adaptiveStep( GA, GAT, hb, m1, u, l, c, z_k_t, z_k_t_new, k, current_omega, current_eta );
         z_bar = ( z_bar * eta_sum + z_k_t_new * current_eta ) / ( eta_sum + current_eta );
         eta_sum += current_eta;

         TNL_ASSERT_TRUE( all( lessEqual( z_bar.getView( 0, n ), u + std::numeric_limits< RealType >::round_error() ) ),
                          "x is not in the feasible region" );
         TNL_ASSERT_TRUE( all( greaterEqual( z_bar.getView( 0, n ), l - std::numeric_limits< RealType >::round_error() ) ),
                          "x is not in the feasible region" );
         TNL_ASSERT_TRUE(
            m1 == 0 || all( greaterEqual( z_bar.getView( n, n + m1 ), -std::numeric_limits< RealType >::round_error() ) ),
            "y is not in the feasible region" );

         RealType error_new, error_bar, error_n_t;
         //bool restart_to_average( false );
         if( true ) {  //t % 1 == 0 ) {
            if( restarting == PDLPRestarting::KKTError ) {
               error_new = KKTError( GA, GAT, m1, c, hb, z_k_t_new, u, l, current_omega );
               error_bar = KKTError( GA, GAT, m1, c, hb, z_bar, u, l, current_omega );
               error_n_0 = KKTError( GA, GAT, m1, c, hb, z_k_0, u, l, current_omega );
               error_n_t = KKTError( GA, GAT, m1, c, hb, z_k_t, u, l, current_omega );
            }
            else if( restarting == PDLPRestarting::DualityGap ) {
               // Solve argmin_{x^hat \in X, y^hat \in Y } [ ((K^T *y )^T - c )*x^hat + ( q - K*x )^T *y^hat ]
               error_new = primalDualGap( GA, GAT, m1, c, hb, u, l, z_k_t_new, z_k_0 );
               error_bar = primalDualGap( GA, GAT, m1, c, hb, u, l, z_bar, z_k_0 );
               error_n_0 = primalDualGap( GA, GAT, m1, c, hb, u, l, z_k_0, last_z );
               error_n_t = primalDualGap( GA, GAT, m1, c, hb, u, l, z_k_t, z_k_0 );
            }
            // Get restart candidate
            z_c = error_new < error_bar ? z_k_t_new : z_bar;

            // Log
            auto new_x_view = z_c.getView( 0, n );
            auto new_y_view = z_c.getView( n, n + m1 + m2 );
            auto [ primal_feasibility, dual_feasibility, primal_objective, dual_objective ] =
               KKT( GA, GAT, m1, c, hb, z_c, u, l );
            const RealType duality_gap = abs( dual_objective - primal_objective );
            const RealType error = duality_gap + primal_feasibility + dual_feasibility;
            std::cout << "ITER: " << std::setw( 6 ) << k << " NORMS=(" << std::setw( 10 ) << l2Norm( new_x_view ) << ", "
                      << std::setw( 10 ) << l2Norm( new_y_view ) << ") INV.STEP : " << std::setw( 10 ) << 1.0 / current_eta
                      << " PRIMAL WEIGHT: " << std::setw( 10 ) << current_omega << " PRIM.OBJ. : " << std::setw( 10 )
                      << primal_objective << " DUAL OBJ. : " << std::setw( 12 ) << dual_objective
                      << " PRIM. FEAS.: " << std::setw( 12 ) << primal_feasibility << " DUAL FEAS.: " << std::setw( 12 )
                      << dual_feasibility << " KKT ERROR: " << std::setw( 10 ) << error << std::endl;
            // End of log

            if( error_new <= beta_sufficient * error_n_0 || ( error_new <= beta_necessary * error_n_0 && error_new > error_n_t )
                || ( t >= beta_artificial * k ) )
            {
               std::cout << "Restart to " << ( error_bar < error_new ? "average" : "current" ) << " after " << t
                         << " iterations." << std::endl;
               break;
            }
         }
         else
            z_c = z_bar;
         z_k_t = z_k_t_new;

         t++;
         k++;
      }
      last_z = z_k_0;
      auto new_x_view = z_c.getView( 0, n );
      auto new_y_view = z_c.getView( n, n + m1 + m2 );

      auto [ primal_feasibility, dual_feasibility, primal_objective, dual_objective ] = KKT( GA, GAT, m1, c, hb, z_c, u, l );

      const RealType epsilon = 1.0e-6;
      const RealType duality_gap = abs( dual_objective - primal_objective );
      const RealType error = duality_gap + primal_feasibility + dual_feasibility;
      if( duality_gap < epsilon * ( abs( dual_objective ) + abs( primal_objective ) )
          && primal_feasibility < epsilon * ( 1 + l2Norm( hb ) ) && dual_feasibility < epsilon * ( 1 + l2Norm( c ) ) )
      {
         //new_x_view *= D2;
         //new_y_view *= D1;
         std::cout << "===============================" << std::endl;
         std::cout << "SOLUTION FOUND" << std::endl;
         std::cout << "KKT ERROR: " << error << std::endl;
         std::cout << "PRIMAL OBJECTIVE: " << primal_objective << std::endl;
         std::cout << "DUAL OBJECTIVE: " << dual_objective << std::endl;
         std::cout << "DUALITY GAP: " << duality_gap << std::endl;
         std::cout << "PRIMAL FEASIBILITY: " << primal_feasibility << std::endl;
         std::cout << "DUAL FEASIBILITY: " << dual_feasibility << std::endl;
         std::cout << "X: " << new_x_view << std::endl;
         std::cout << "D2 * X: " << D2 * new_x_view << std::endl;
         std::cout << "Y: " << new_y_view << std::endl;
         x = new_x_view;
         //y = new_y_view;
         return { true, dual_objective, error };
      }
      /*else
         std::cout << "ITER: " << std::setw( 6 ) << k << " NORMS=(" << std::setw( 10 ) << l2Norm( new_x_view ) << ", "
                   << std::setw( 10 ) << l2Norm( new_y_view ) << ") INV.STEP : " << std::setw( 10 ) << 1.0 / current_eta
                   << " PRIMAL WEIGHT: " << std::setw( 10 ) << current_omega << " PRIM.OBJ. : " << std::setw( 10 )
                   << primal_objective << " DUAL OBJ. : " << std::setw( 12 ) << dual_objective
                   << " PRIM. FEAS.: " << std::setw( 12 ) << primal_feasibility << " DUAL FEAS.: " << std::setw( 12 )
                   << dual_feasibility << " KKT ERROR: " << std::setw( 10 ) << error << std::endl;*/

      //Compute new parameter omega
      RealType delta_x = lpNorm( new_x_view - x, 2 );
      RealType delta_y = lpNorm( new_y_view - y, 2 );
      if( delta_x > 1.0e-10 && delta_y > 1.0e-10 ) {
         const RealType theta = 0.5;
         current_omega = exp( theta * log( delta_y / delta_x ) + ( 1.0 - theta ) * log( current_omega ) );
      }
   }
   return { false, 0.0, 0.0 };
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::adaptiveStep( const MatrixType& GA,
                                                 const MatrixType& GAT,
                                                 const VectorType& hb,
                                                 const IndexType m1,
                                                 const VectorType& u,
                                                 const VectorType& l,
                                                 const VectorType& c,
                                                 const VectorType& in_z,
                                                 VectorType& out_z,
                                                 const IndexType k,
                                                 RealType& current_omega,
                                                 RealType& current_eta )
{
   const IndexType m = GA.getRows();
   const IndexType m2 = m - m1;
   const IndexType n = GA.getColumns();
   const IndexType N = n + m;

   auto in_x = in_z.getConstView( 0, n );
   auto in_y = in_z.getConstView( n, N );
   auto out_x = out_z.getView( 0, n );
   auto out_y = out_z.getView( n, N );

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

   //std::cout << "GA values:" << GA.getValues() << std::endl;
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

      // Compute new parameter eta
      delta_x = out_x - in_x;
      delta_y = out_y - in_y;
      const RealType delta_z_norm = current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega;

      GA.vectorProduct( delta_x, Kx );
      const RealType div = 2.0 * abs( ( Kx, delta_y ) );  // TODO: It is 0.5 in source code
      const RealType max_eta = div > 0 ? delta_z_norm / div : std::numeric_limits< RealType >::infinity();
      const RealType new_eta = min( ( 1.0 - pow( k + 1, -0.3 ) ) * max_eta, ( 1.0 + pow( k + 1, -0.6 ) ) * current_eta );
      TNL_ASSERT_GT( new_eta, 0, "new_eta <= 0" );
      if( new_eta < max_eta ) {
         current_eta = new_eta;
         return;
      }
      else
         current_eta = new_eta;
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

   VectorType z_hat( N, 0 );
   auto z_hat_view = z_hat.getView();
   RealType r = max( 0.001, l2Norm( z_ref - z ) );  // TODO: How to deal with small r?
   if( ! linearTrustRegion( z, g_l.getView(), g_u.getView(), g.getView(), r, z_hat_view ) )
      throw std::runtime_error( "linearTrustRegion failed" );
   return ( ( c, x_view ) - ( y_view, q ) - ( z_hat, g ) ) / r;  // TODO: How to deal with small r?
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::KKT( const MatrixType& GA,
                                        const MatrixType& GAT,
                                        const IndexType m1,
                                        const VectorType& c,
                                        const VectorType& q,
                                        const VectorView& z,
                                        const VectorType& u,
                                        const VectorType& l ) const -> std::tuple< RealType, RealType, RealType, RealType >
{
   const IndexType m = GA.getRows();
   const IndexType n = GA.getColumns();
   const IndexType N = n + m;

   auto x = z.getConstView( 0, n );
   auto y = z.getConstView( n, N );
   auto c_view = c.getConstView();

   // Compute error of the primal feasibility
   VectorType aux( m );
   if( m1 > 0 ) {
      auto aux1 = aux.getView( 0, m1 );
      GA.vectorProduct( x, aux1, 1, 0, 0, m1 );  // aux1 = G * x
      auto h = q.getConstView( 0, m1 );
      aux1 = maximum( h - aux1, 0 );
   }
   if( m > m1 ) {
      auto aux2 = aux.getView( m1, m );
      GA.vectorProduct( x, aux2, 1, 0, m1, m );  // aux2 = A * x
      auto b = q.getConstView( m1, m );
      aux2 -= b;
   }
   const RealType primal_feasibility = l2Norm( aux );

   // Compute error of the dual feasibility
   VectorType KTy( n ), lambda( n );
   GAT.vectorProduct( y, KTy );
   auto KTy_view = KTy.getConstView();
   auto l_view = l.getConstView();
   auto u_view = u.getConstView();
   lambda.forAllElements(
      [ = ] __cuda_callable__( IndexType i, RealType & value )
      {
         if( l_view[ i ] == -std::numeric_limits< RealType >::infinity()
             && u_view[ i ] == std::numeric_limits< RealType >::infinity() )
            value = 0;
         else {
            value = c_view[ i ] - KTy_view[ i ];
            if( l_view[ i ] == -std::numeric_limits< RealType >::infinity() ) {
               value = min( value, 0 );
            }
            else if( u_view[ i ] == std::numeric_limits< RealType >::infinity() ) {
               value = max( value, 0 );
            }
         }
      } );
   //std::cout << "c - KTy = " << c_view - KTy_view << " lambda = " << lambda << " l = " << l << " u = " << u << std::endl;
   const RealType dual_feasibility = l2Norm( c - KTy - lambda );

   // Compute the primal objective
   const RealType primal_objective = ( c, x );

   // Compute the dual objective
   auto lambda_view = lambda.getConstView();
   const RealType dual_objective =
      ( q, y )
      + Algorithms::reduce< DeviceType >( (IndexType) 0,
                                          n,
                                          [ = ] __cuda_callable__( IndexType i ) -> RealType
                                          {
                                             RealType result = 0;
                                             if( l_view[ i ] != -std::numeric_limits< RealType >::infinity() )
                                                result += l_view[ i ] * max( lambda_view[ i ], 0 );
                                             if( u_view[ i ] != std::numeric_limits< RealType >::infinity() )
                                                result -= u_view[ i ] * min( lambda_view[ i ], 0 );
                                             return result;
                                          },
                                          TNL::Plus{} );

   return { primal_feasibility, dual_feasibility, primal_objective, dual_objective };
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::KKTError( const MatrixType& GA,
                                             const MatrixType& GAT,
                                             const IndexType m1,
                                             const VectorType& c,
                                             const VectorType& q,
                                             const VectorView& z,
                                             const VectorType& u,
                                             const VectorType& l,
                                             const RealType& omega ) const -> RealType
{
   auto [ primal_feasibility, dual_feasibility, primal_objective, dual_objective ] = KKT( GA, GAT, m1, c, q, z, u, l );
   const RealType omega_sqr = omega * omega;
   return sqrt( omega_sqr * primal_feasibility * primal_feasibility + 1.0 / omega_sqr * ( dual_feasibility * dual_feasibility )
                + pow( primal_objective - dual_objective, 2 ) );
}

}  // namespace TNL::Solvers::Optimization
