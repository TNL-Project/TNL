// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Containers/NDArray.h>
#include "PDLP.h"
#include "LinearTrustRegion.h"

#include <iomanip>

namespace TNL::Solvers::Optimization {

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::solve( const LPProblemType& lpProblem, VectorType& x )
   -> std::tuple< bool, RealType, RealType >
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

   // Preconditioning
   /*T.setSize( n );
   Sigma.setSize( m );
   T = 1;
   Sigma = 1;
   auto T_view = T.getView();
   auto Sigma_view = Sigma.getView();
   const RealType alfa = 1.0;
   GA.reduceAllRows(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return pow( abs( value ), 2.0 - alfa );
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         T_view[ rowIdx ] = 1.0 / value;
      },
      0.0 );
   GAT.reduceAllRows(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return pow( abs( value ), alfa );
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         T_view[ rowIdx ] = 1.0 / value;
      },
      0.0 );
   std::cout << "T = " << T << " Sigma = " << Sigma << std::endl;*/

   VectorType y( m1 + m2, 0 );
   const RealType max_norm = Matrices::maxNorm( GA );
   if( max_norm < 1.0e-10 )
      throw std::runtime_error( "Matrix for the LP problem is nearly zero matrix." );
   const RealType initial_eta = 1.0 / max_norm;
   const RealType c_norm = lpNorm( c, 2 );
   const RealType q_norm = l2Norm( hb );
   const RealType initial_omega = ( c_norm > 1.0e-10 && q_norm > 1.0e-10 ) ? c_norm / q_norm : 1;

   if( restarting == PDLPRestarting::KKTError )
      this->primal_gradient.setSize( n );
   IndexType k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;

   const IndexType max_restarting_steps = 50;
   Array2D z_container;
   z_container.setSizes( max_restarting_steps + 1, n + m1 + m2 );
   auto z_container_view = z_container.getView();
   VectorType z_c( N ), z_bar( N ), last_z( N ), eta_container( max_restarting_steps + 1 );
   auto z_bar_view = z_bar.getView();
   z_bar_view = 0;
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
   last_z = z_c;
   x = z_c.getView( 0, n );
   y = z_c.getView( n, N );
   auto eta_container_view = eta_container.getView();
   RealType eta_sum( 0 ), error_n_0( 0 );

   VectorType z_k_0( N );
   while( k < 1000000 ) {  //this->nextIteration() ) {
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
         adaptiveStep( GA, GAT, hb, m1, u, l, c, in_x_view, in_y_view, out_x_view, out_y_view, k, current_omega, current_eta );
         eta_container[ t + 1 ] = current_eta;
         //z_bar = ( z_bar * eta_sum + z_new_view * current_eta ) / ( eta_sum + current_eta );
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
         TNL_ASSERT_TRUE( all( lessEqual( z_bar.getView( 0, n ), u + std::numeric_limits< RealType >::round_error() ) ),
                          "x is not in the feasible region" );
         TNL_ASSERT_TRUE( all( greaterEqual( z_bar.getView( 0, n ), l - std::numeric_limits< RealType >::round_error() ) ),
                          "x is not in the feasible region" );
         TNL_ASSERT_TRUE(
            m1 == 0 || all( greaterEqual( z_bar.getView( n, n + m1 ), -std::numeric_limits< RealType >::round_error() ) ),
            "y is not in the feasible region" );
         t++;
         k++;

         RealType error_new, error_bar, error_n_t;
         VectorView z_n_t_view( &z_container( t, 0 ), N );
         if( t % 1 == 0 ) {
            if( restarting == PDLPRestarting::KKTError ) {
               error_new = KKTError( GA, GAT, m1, c, hb, z_new_view, u, l, current_omega );
               error_bar = KKTError( GA, GAT, m1, c, hb, z_bar_view, u, l, current_omega );
               error_n_0 = KKTError( GA, GAT, m1, c, hb, z_view, u, l, current_omega );
               error_n_t = KKTError( GA, GAT, m1, c, hb, z_n_t_view, u, l, current_omega );
            }
            else if( restarting == PDLPRestarting::DualityGap ) {
               // Solve argmin_{x^hat \in X, y^hat \in Y } [ ((K^T *y )^T - c )*x^hat + ( q - K*x )^T *y^hat ]
               error_new = primalDualGap( GA, GAT, m1, c, hb, u, l, z_new_view, z_view );
               error_bar = primalDualGap( GA, GAT, m1, c, hb, u, l, z_bar_view, z_view );
               error_n_0 = primalDualGap( GA, GAT, m1, c, hb, u, l, z_view, last_z );
               error_n_t = primalDualGap( GA, GAT, m1, c, hb, u, l, z_n_t_view, z_view );
            }
            // Get restart candidate
            z_c = error_new < error_bar ? z_new_view : z_bar_view;
            if( error_new <= beta_sufficient * error_n_0 || ( error_new <= beta_necessary * error_n_0 && error_new > error_n_t )
                || ( t >= beta_artificial * k ) )
               break;
         }
         else
            z_c = z_bar_view;
      }
      last_z = z_view;
      auto new_x_view = z_c.getView( 0, n );
      auto new_y_view = z_c.getView( n, n + m1 + m2 );

      auto [ primal_feasibility, dual_feasibility, primal_objective, dual_objective ] = KKT( GA, GAT, m1, c, hb, z_c, u, l );

      const RealType epsilon = 1.0e-4;
      const RealType duality_gap = abs( dual_objective - primal_objective );
      const RealType error = duality_gap + primal_feasibility + dual_feasibility;
      if( duality_gap < epsilon * ( abs( dual_objective ) + abs( primal_objective ) )
          && primal_feasibility < epsilon * ( 1 + l2Norm( hb ) ) && dual_feasibility < epsilon * ( 1 + l2Norm( c ) ) )
      {
         std::cout << "===============================" << std::endl;
         std::cout << "SOLUTION FOUND" << std::endl;
         std::cout << "ERROR: " << error << std::endl;
         std::cout << "PRIMAL OBJECTIVE: " << primal_objective << std::endl;
         std::cout << "DUAL OBJECTIVE: " << dual_objective << std::endl;
         std::cout << "DUALITY GAP: " << duality_gap << std::endl;
         std::cout << "PRIMAL FEASIBILITY: " << primal_feasibility << std::endl;
         std::cout << "DUAL FEASIBILITY: " << dual_feasibility << std::endl;
         std::cout << "X: " << new_x_view << std::endl;
         std::cout << "Y: " << new_y_view << std::endl;
         return { true, dual_objective, error };
      }
      else
         std::cout << "ITER: " << std::setw( 6 ) << k << " PRIM. OBJ.: " << std::setw( 10 ) << primal_objective
                   << " DUAL OBJ.: " << std::setw( 12 ) << dual_objective << " PRIM. FEAS.: " << std::setw( 12 )
                   << primal_feasibility << " DUAL FEAS.: " << std::setw( 12 ) << dual_feasibility
                   << " ERROR: " << std::setw( 10 ) << error
                   << std::endl;  // << " X: " << new_x_view << " Y: " << new_y_view << std::endl;

      //Compute new parameter omega
      RealType delta_x = lpNorm( new_x_view - x, 2 );
      RealType delta_y = lpNorm( new_y_view - y, 2 );
      if( delta_x > 1.0e-10 && delta_y > 1.0e-10 ) {
         const RealType theta = 0.5;
         current_omega = exp( theta * log( delta_y / delta_x ) + ( 1.0 - theta ) * log( current_omega ) );
      }
      std::cout << "OMEGA: " << current_omega << std::endl;
      x = new_x_view;
      y = new_y_view;
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

      // Compute new parameter eta
      delta_x = out_x - in_x;
      delta_y = out_y - in_y;
      const RealType delta_z_norm = sqrt( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      GA.vectorProduct( delta_x, Kx );
      const RealType div = 2.0 * abs( ( Kx, delta_y ) );
      if( abs( div ) > 1.0e-10 ) {
         const RealType max_eta = delta_z_norm / div;
         const RealType new_eta = min( ( 1.0 - pow( k + 1, -0.3 ) ) * max_eta, ( 1.0 + pow( k + 1, -0.6 ) ) * current_eta );
         if( new_eta < max_eta ) {
            current_eta = new_eta;
            return;
         }
         else
            current_eta = new_eta;
      }
      else {
         current_eta = 1.0 / max_norm;
         return;
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
