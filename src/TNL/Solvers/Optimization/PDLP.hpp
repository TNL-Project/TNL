// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/MatrixWriter.h>
#include "PDLP.h"
#include "LinearTrustRegion.h"
#include "Preconditioning/PockChambolle.h"

#include <iomanip>

//#define PRINTING

namespace TNL::Solvers::Optimization {

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::configSetup( Config::ConfigDescription& config, const std::string& prefix )
{
   IterativeSolver< RealType, IndexType, SolverMonitor >::configSetup( config, prefix );
   config.addEntry< bool >( prefix + "inequalities-first",
                            "The first rows of the constraint matrix are just inequalities, equalities are in the rest.",
                            true );
   config.addEntry< std::string >( prefix + "restarting", "Restarting strategy.", "kkt" );
   config.addEntryEnum( "none" );
   config.addEntryEnum( "constant" );
   config.addEntryEnum( "duality-gap" );
   config.addEntryEnum( "kkt" );
   config.addEntryEnum( "fast" );
   config.addEntry< int >(
      prefix + "max-restarting-interval", "Maximum interval without restarting interval. Zero means no limit.", 0 );
   config.addEntry< bool >( prefix + "write-convergence-graphs", "Write convergence graphs for the solver.", false );
}

template< typename LPProblem_, typename SolverMonitor >
bool
PDLP< LPProblem_, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const std::string& prefix )
{
   this->setInequalitiesFirst( parameters.getParameter< bool >( prefix + "inequalities-first" ) );
   auto restarting = parameters.getParameter< std::string >( prefix + "restarting" );
   if( restarting == "none" )
      this->setRestarting( PDLPRestarting::None );
   else if( restarting == "constant" )
      this->setRestarting( PDLPRestarting::Constant );
   else if( restarting == "duality-gap" )
      this->setRestarting( PDLPRestarting::DualityGap );
   else if( restarting == "kkt" )
      this->setRestarting( PDLPRestarting::KKT );
   else if( restarting == "fast" )
      this->setRestarting( PDLPRestarting::Fast );
   else
      throw std::runtime_error( "Unknown restarting strategy: " + restarting );
   auto restartingInterval = parameters.getParameter< int >( prefix + "max-restarting-interval" );
   if( restartingInterval < 0 )
      throw std::runtime_error( "Restarting interval must be non-negative." );
   this->setMaximalRestartingInterval( restartingInterval );
   this->writeConvergenceGraphs = parameters.getParameter< bool >( prefix + "write-convergence-graphs" );

   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::solve( const LPProblemType& lpProblem, VectorType& x )
   -> std::tuple< bool, RealType, RealType >
{
   //this->segmentsReductionKernel.setThreadsMapping( Algorithms::SegmentsReductionKernels::LightCSRConstantThreads );
   //this->segmentsReductionKernel.setThreadsPerSegment( 128 );
   solverTimer.start();
   this->K = lpProblem.getConstraintMatrix();
   this->c = lpProblem.getObjectiveFunction();
   this->q = lpProblem.getConstraintVector();
   this->l = lpProblem.getLowerBounds();
   this->u = lpProblem.getUpperBounds();
   this->inequalitiesFirst = lpProblem.getInequalitiesFirst();

   this->m1 = lpProblem.getInequalityCount();  // TODO: Rename the method
   this->m = K.getRows();
   this->m2 = m - m1;
   this->n = K.getColumns();
   this->N = n + m;

   TNL_ASSERT_EQ( c.getSize(), n, "" );
   TNL_ASSERT_EQ( q.getSize(), m, "" );
   TNL_ASSERT_EQ( l.getSize(), n, "" );
   TNL_ASSERT_EQ( u.getSize(), n, "" );
   TNL_ASSERT_EQ( x.getSize(), n, "" );
   this->KT.getTransposition( K );

   /*Matrices::MatrixWriter< MatrixType >::writeEps( "K.eps", K );
   Matrices::MatrixWriter< MatrixType >::writeEps( "KT.eps", KT );
   Matrices::MatrixWriter< MatrixType >::writeMtx( "K.mtx", K );
   Matrices::MatrixWriter< MatrixType >::writeMtx( "KT.mtx", KT );*/

   // Exporting bounds
   /*std::fstream file( "lower-bounds.txt", std::ios::out );
   if( file.is_open() ) {
      for( IndexType i = 0; i < l.getSize(); ++i ) {
         file << l.getElement( i ) << std::endl;
      }
      file.close();
   }
   file.open( "upper-bounds.txt", std::ios::out );
   if( file.is_open() ) {
      for( IndexType i = 0; i < u.getSize(); ++i ) {
         file << u.getElement( i ) << std::endl;
      }
      file.close();
   }*/

   // Filter the bounds
   this->filtered_l = this->l;
   this->filtered_u = this->u;
   this->filtered_l.forAllElements(
      [ = ] __cuda_callable__( IndexType i, RealType & value ) mutable
      {
         if( value == -std::numeric_limits< RealType >::infinity() )
            value = 0;
      } );
   this->filtered_u.forAllElements(
      [ = ] __cuda_callable__( IndexType i, RealType & value ) mutable
      {
         if( value == std::numeric_limits< RealType >::infinity() )
            value = 0;
      } );

   // Preconditioning
   D1.setSize( m );
   D2.setSize( n );
   D1 = 1;
   D2 = 1;
   //Preconditioning::pockChambole( K, KT, m1, c, q, l, u, D1, D2 );
   //std::cout << "D1: " << D1 << std::endl;
   //std::cout << "D2: " << D2 << std::endl;

   VectorType y( m, 0 );  // TODO: This should argument maybe

   this->Kx.setSize( m );
   this->Kx_new.setSize( m );
   this->Kx_averaged.setSize( m );
   this->KTy.setSize( n );
   this->KTy_averaged.setSize( n );
   this->lambda.setSize( n );
   //this->K_norm = Matrices::spectralNorm( K, KT );
   //std::cout << "Constraint matrix spectral norm: " << this->K_norm << std::endl;

   std::cout << "Columns:           " << n << std::endl;
   std::cout << "Rows:              " << m << std::endl;
   std::cout << "C norm:            " << l2Norm( c ) << std::endl;
   std::cout << "Q norm:            " << l2Norm( q ) << std::endl;
   std::cout << "Lower bounds norm: " << l2Norm( this->filtered_l ) << std::endl;
   std::cout << "Upper bounds norm: " << l2Norm( this->filtered_u ) << std::endl;
   //std::cout << "Initial omega:     " << initial_omega << std::endl;
   //std::cout << "Initial eta:       " << initial_eta << std::endl;

#ifdef PRINTING
   std::cout << std::setprecision( 16 );
   /*std::cout << "q = " << q << std::endl;
   std::cout << "c = " << c << std::endl;
   std::cout << "l = " << l << std::endl;
   std::cout << "u = " << u << std::endl;*/
#endif

   auto l_view = l.getConstView();
   auto u_view = u.getConstView();
   x.forElements( 0,
                  n,
                  [ = ] __cuda_callable__( IndexType i, RealType & value ) mutable
                  {
                     if( l_view[ i ] == -std::numeric_limits< RealType >::infinity() )
                        value = min( u_view[ i ], 0 );
                     else
                        value = l_view[ i ];
                  } );
   y = 0;

   if( writeConvergenceGraphs ) {
      kkt_current_primal_objective_file.open( "kkt-current-primal-objective.txt", std::ios::out );
      kkt_current_dual_objective_file.open( "kkt-current-dual-objective.txt", std::ios::out );
      kkt_averaged_primal_objective_file.open( "kkt-averaged-primal-objective.txt", std::ios::out );
      kkt_averaged_dual_objective_file.open( "kkt-averaged-dual-objective.txt", std::ios::out );
      kkt_current_duality_gap_file.open( "kkt-current-duality-gap.txt", std::ios::out );
      kkt_averaged_duality_gap_file.open( "kkt-averaged-duality-gap.txt", std::ios::out );
      kkt_current_primal_feasibility_file.open( "kkt-current-primal-feasibility.txt", std::ios::out );
      kkt_current_dual_feasibility_file.open( "kkt-current-dual-feasibility.txt", std::ios::out );
      kkt_averaged_primal_feasibility_file.open( "kkt-averaged-primal-feasibility.txt", std::ios::out );
      kkt_averaged_dual_feasibility_file.open( "kkt-averaged-dual-feasibility.txt", std::ios::out );
      kkt_current_mu_file.open( "kkt-current-mu.txt", std::ios::out );
      kkt_averaged_mu_file.open( "kkt-averaged-mu.txt", std::ios::out );
      fast_current_primal_objective_file.open( "fast-current-primal-objective.txt", std::ios::out );
      fast_current_dual_objective_file.open( "fast-current-dual-objective.txt", std::ios::out );
      fast_averaged_primal_objective_file.open( "fast-averaged-primal-objective.txt", std::ios::out );
      fast_averaged_dual_objective_file.open( "fast-averaged-dual-objective.txt", std::ios::out );
      fast_current_duality_gap_file.open( "fast-current-duality-gap.txt", std::ios::out );
      fast_averaged_duality_gap_file.open( "fast-averaged-duality-gap.txt", std::ios::out );
      fast_current_primal_feasibility_file.open( "fast-current-primal-feasibility.txt", std::ios::out );
      fast_current_dual_feasibility_file.open( "fast-current-dual-feasibility.txt", std::ios::out );
      fast_current_mu_file.open( "fast-current-mu.txt", std::ios::out );
      fast_averaged_mu_file.open( "fast-averaged-mu.txt", std::ios::out );
      current_gradient_file.open( "current-gradient.txt", std::ios::out );
      averaged_gradient_file.open( "averaged-gradient.txt", std::ios::out );

      restarts_file.open( "restarts.txt", std::ios::out );
   }

   spmvTimer.reset();
   return PDHG( x, y );
}
template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::PDHG( VectorType& x, VectorType& y ) -> std::tuple< bool, RealType, RealType >
{
   this->K_norm = max( abs( K.getValues() ) );
   if( this->K_norm < 1.0e-10 )
      throw std::runtime_error( "Matrix for the LP problem is nearly zero matrix." );
   const RealType initial_eta = 1.0 / this->K_norm;
   const RealType c_norm = l2Norm( c );
   const RealType q_norm = l2Norm( q );
   const RealType initial_omega = ( c_norm > 1.0e-10 && q_norm > 1.0e-10 ) ? c_norm / q_norm : 1;

   auto Kx_view = Kx.getView();
   computeKx( x, Kx_view );
   auto KTy_view = KTy.getView();
   computeKTy( y, KTy_view );
   KKTDataType kkt_candidate, kkt_last_restart;

   IndexType k = 0;
   this->adaptive_k = 1;
   RealType current_eta = initial_eta;
   RealType current_omega = initial_omega;
   this->KxComputations = 0;
   this->KTyComputations = 0;

   VectorType z_candidate( N ), z_averaged( N ), z_last_restart( N ), z_last_iteration( N ), z_current( N );
   z_candidate.getView( 0, n ) = x;
   z_candidate.getView( n, N ) = y;

   RealType eta_sum( 0 ), mu_last_restart( std::numeric_limits< RealType >::infinity() ), mu_candidate( 0 ),
      mu_last_candidate( 0 );

   while( k < this->getMaxIterations() ) {
      IndexType t = 0;

      eta_sum = 0;
      z_last_iteration = z_last_restart = z_averaged = z_candidate;
      if( ! this->averaging ) {
         adaptiveStep( z_last_iteration, z_candidate, k, current_omega, current_eta );
         k++;
         z_last_iteration = z_candidate;
      }
      else {
         mu_last_candidate = mu_last_restart;
         while( k < this->getMaxIterations() ) {
            adaptiveStep( z_last_iteration, z_current, k, current_omega, current_eta );
            z_averaged = ( z_averaged * eta_sum + z_current * current_eta ) / ( eta_sum + current_eta );
            k++;
            t++;

#ifdef PRINTING
            std::cout << "xUpdate = " << l2Norm( z_current.getView( 0, n ) ) << std::endl;
            std::cout << "yUpdate = " << l2Norm( z_current.getView( n, n + m1 + m2 ) ) << std::endl;
            std::cout << "xAverage = " << l2Norm( z_averaged.getView( 0, n ) ) << std::endl;
            std::cout << "yAverage = " << l2Norm( z_averaged.getView( n, n + m1 + m2 ) ) << std::endl;
            std::cout << "eta sum " << eta_sum << " -> " << eta_sum + current_eta << "(adding " << current_eta << ") "
                      << std::endl;
            std::cout << "Scaling by " << 1.0 / ( eta_sum + current_eta ) << std::endl;
#endif
            eta_sum += current_eta;

            TNL_ASSERT_TRUE( all( lessEqual( z_averaged.getView( 0, n ), u + std::numeric_limits< RealType >::round_error() ) ),
                             "x is not in the feasible region" );
            TNL_ASSERT_TRUE(
               all( greaterEqual( z_averaged.getView( 0, n ), l - std::numeric_limits< RealType >::round_error() ) ),
               "x is not in the feasible region" );
            TNL_ASSERT_TRUE(
               ! this->inequalitiesFirst || m1 == 0
                  || all( greaterEqual( z_averaged.getView( n, n + m1 ), -std::numeric_limits< RealType >::round_error() ) ),
               "y is not in the feasible region" );
            TNL_ASSERT_TRUE(
               this->inequalitiesFirst || m2 == 0
                  || all( greaterEqual( z_averaged.getView( n + m1, N ), -std::numeric_limits< RealType >::round_error() ) ),
               "y is not in the feasible region" );

            // TODO: Return back into the fast restarting if branch
            RealType current_primal_objective = ( c, z_current.getView( 0, n ) );
            computeLambda( c, KTy, l, u, lambda );
            const RealType current_dual_objective =
               ( q, z_current.getView( n, N ) ) + ( filtered_l, maximum( lambda, 0 ) ) + ( filtered_u, minimum( lambda, 0 ) );
            const RealType current_duality_gap = current_primal_objective - current_dual_objective;
            const RealType current_primal_feasibility = computePrimalFeasibility( q, Kx );
            const RealType current_dual_feasibility = l2Norm( c - KTy - lambda );
            const RealType current_omega_sqrt = current_omega * current_omega;
            const RealType mu_current = sqrt( current_omega_sqrt * current_primal_feasibility * current_primal_feasibility
                                              + current_dual_feasibility * current_dual_feasibility / current_omega_sqrt
                                              + current_duality_gap * current_duality_gap );

            RealType averaged_primal_objective = ( c, z_averaged.getView( 0, n ) );
            //KT.vectorProduct( z_averaged.getView( n, N ), KTy, segmentsReductionKernel );
            //this->matrixVectorProducts++;
            //computeLambda( c, KTy, l, u, lambda );
            const RealType averaged_dual_objective =
               ( q, z_averaged.getView( n, N ) ) + ( filtered_l, maximum( lambda, 0 ) ) + ( filtered_u, minimum( lambda, 0 ) );
            const RealType averaged_duality_gap = averaged_primal_objective - averaged_dual_objective;
            const RealType mu_averaged = sqrt( current_omega_sqrt * current_primal_feasibility * current_primal_feasibility
                                               + current_dual_feasibility * current_dual_feasibility / current_omega_sqrt
                                               + averaged_duality_gap * averaged_duality_gap );

            if( writeConvergenceGraphs ) {
               fast_current_primal_objective_file << k << " " << current_primal_objective << std::endl;
               fast_current_dual_objective_file << k << " " << current_dual_objective << std::endl;
               fast_averaged_primal_objective_file << k << " " << averaged_primal_objective << std::endl;
               fast_averaged_dual_objective_file << k << " " << averaged_dual_objective << std::endl;
               fast_current_duality_gap_file << k << " " << mu_current << std::endl;
               fast_averaged_duality_gap_file << k << " " << mu_averaged << std::endl;
               fast_current_primal_feasibility_file << k << " " << current_primal_feasibility << std::endl;
               fast_current_dual_feasibility_file << k << " " << current_dual_feasibility << std::endl;
               fast_current_mu_file << k << " " << mu_current << std::endl;
               fast_averaged_mu_file << k << " " << mu_averaged << std::endl;
            }

            if( restarting == PDLPRestarting::Fast ) {
               if( mu_averaged < mu_current ) {
                  mu_candidate = mu_averaged;
                  z_candidate = z_averaged;
               }
               else {
                  mu_candidate = mu_current;
                  z_candidate = z_current;
               }
               //const RealType mu_averaged = std::numeric_limits< RealType >::infinity();
               mu_candidate = mu_current;
               z_candidate = z_current;
               //if( k % 10 == 1 ) {
               //   kkt_candidate = KKT( z_candidate );
               //}

               if( t >= beta_artificial * k ) {
                  std::cout << "ARTIFICIAL FAST restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  break;
               }
               if( mu_candidate <= beta_sufficient * mu_last_restart ) {
                  std::cout << "SUFFICIENT FAST restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  break;
               }
               if( mu_candidate <= beta_necessary * mu_last_restart && mu_candidate > mu_last_candidate ) {
                  std::cout << "NECESSARY FAST restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  break;
               }
               /*const RealType update = maxNorm( z_candidate - z_last_iteration );
               if( update < 0.00001 * mu_candidate ) {
                  std::cout << "GRADIENT FAST restart to AVERAGE at k = " << k << " t = " << t << " update = " << update
                            << " gap = " << mu_candidate << std::endl;
                  mu_last_restart = mu_candidate;
                  break;
               }*/
            }
            else if( restarting != PDLPRestarting::None ) {
               RealType mu_current, mu_averaged;
               KKTDataType kkt_current, kkt_averaged;
               if( restarting == PDLPRestarting::KKT || restarting == PDLPRestarting::Constant ) {
                  //std::cout << "KKT for current: ";
                  //auto Kx_view = Kx.getView();
                  auto KTy_view = KTy.getView();
                  //computeKx( z_current.getView( 0, n ), Kx_view );
                  computeKTy( z_current.getView( n, N ), KTy_view );
                  kkt_current = KKT( z_current, Kx, KTy );
                  mu_current = kkt_current.getKKTError( current_omega );
                  //std::cout << "KKT for average:";

                  auto KTy_averaged_view = KTy_averaged.getView();
                  auto Kx_averaged_view = Kx_averaged.getView();
                  computeKx( z_averaged.getConstView( 0, n ), Kx_averaged_view );
                  computeKTy( z_averaged.getConstView( n, N ), KTy_averaged_view );
                  kkt_averaged = KKT( z_averaged.getView(), Kx_averaged, KTy_averaged );
                  mu_averaged = kkt_averaged.getKKTError( current_omega );
               }
               else if( restarting == PDLPRestarting::DualityGap ) {
                  mu_current = primalDualGap( z_current, z_last_restart );
                  mu_averaged = primalDualGap( z_averaged, z_last_restart );
               }

               // Get restart candidate
               if( mu_current <= mu_averaged ) {
                  z_candidate = z_current;
                  mu_candidate = mu_current;
                  kkt_candidate = kkt_current;
               }
               else {
                  z_candidate = z_averaged;
                  mu_candidate = mu_averaged;
                  kkt_candidate = kkt_averaged;
                  Kx = Kx_averaged;
                  KTy = KTy_averaged;
               }

               if( this->maxRestartingInterval > 0 && t % this->maxRestartingInterval == 0 ) {
                  std::cout << "CONSTANT restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  kkt_last_restart = kkt_candidate;
                  break;
               }

               if( writeConvergenceGraphs ) {
                  const RealType gr_rst_current_update = maxNorm( z_current - z_last_iteration );
                  const RealType gr_rst_averaged_update = maxNorm( z_averaged - z_last_iteration );
                  const RealType gr_rst_current_duality_gap = kkt_current.getDualityGap();
                  const RealType gr_rst_averaged_duality_gap = kkt_averaged.getDualityGap();
                  const RealType gr_rst_candidate_update = maxNorm( z_candidate - z_last_iteration );
                  const RealType gr_rst_candidate_duality_gap = kkt_candidate.getDualityGap();

                  kkt_current_primal_objective_file << k << " " << kkt_current.getPrimalObjective() << std::endl;
                  kkt_current_dual_objective_file << k << " " << kkt_current.getDualObjective() << std::endl;
                  kkt_averaged_primal_objective_file << k << " " << kkt_averaged.getPrimalObjective() << std::endl;
                  kkt_averaged_dual_objective_file << k << " " << kkt_averaged.getDualObjective() << std::endl;
                  kkt_current_duality_gap_file << k << " " << kkt_current.getDualityGap() << std::endl;
                  kkt_averaged_duality_gap_file << k << " " << kkt_averaged.getDualityGap() << std::endl;
                  kkt_current_primal_feasibility_file << k << " " << kkt_current.getPrimalFeasibility() << std::endl;
                  kkt_current_dual_feasibility_file << k << " " << kkt_current.getDualFeasibility() << std::endl;
                  kkt_averaged_primal_feasibility_file << k << " " << kkt_averaged.getPrimalFeasibility() << std::endl;
                  kkt_averaged_dual_feasibility_file << k << " " << kkt_averaged.getDualFeasibility() << std::endl;
                  kkt_current_mu_file << k << " " << mu_current << std::endl;
                  kkt_averaged_mu_file << k << " " << mu_averaged << std::endl;
                  current_gradient_file << k << " " << gr_rst_current_update << std::endl;
                  averaged_gradient_file << k << " " << gr_rst_averaged_update << std::endl;
               }
#ifdef PRINTING
               std::cout << "k = " << k << " t = " << t << std::endl;
               std::cout << "Restarting errs.: current = " << mu_current << " average = " << mu_averaged << std::endl;
               std::cout << "Artificial test: " << t << " >= " << beta_artificial * k << std::endl;
               //printf( "Artificial test: %d >= %g * %d = %g\n", t, beta_artificial, k, beta_artificial * k );
#endif
               if( t >= beta_artificial * k ) {
                  std::cout << "ARTIFICIAL restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  kkt_last_restart = kkt_candidate;
                  if( writeConvergenceGraphs )
                     restarts_file << k << " ARTIFICIAL " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" ) << std::endl;
                  break;
               }

               // Kuba restarting
               // "maly subgradient a maly gap, restart probability treba 10e^-8;
               // maly subgradient a velky gap, restart probability close to 1;
               // jinak neco mezi, treba 10^-4"

               //const RealType beta_gap = 100;
               /*if( gr_rst_duality_gap > beta_gap
                   //|| kkt_candidate.getPrimalFeasibility() > beta_gap
                   //|| kkt_candidate.getDualFeasibility() > beta_gap
                   && gr_rst_update < 0.1 )
               {
                  std::cout << "GRADIENT restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  std::cout << "   Duality gap: " << kkt_candidate.getDualityGap() << std::endl;
                  std::cout << "   Primal feas.: " << kkt_candidate.getPrimalFeasibility() << std::endl;
                  std::cout << "   Dual feas.: " << kkt_candidate.getDualFeasibility() << std::endl;
                  std::cout << "   Update: " << gr_rst_update << std::endl;
                  mu_last_restart = mu_candidate;
                  kkt_last_restart = kkt_candidate;
                  break;
               }*/
               //if( k % 25 == 0 )
               //   break;

               if( restarting == PDLPRestarting::KKT ) {
                  mu_last_restart = kkt_last_restart.getKKTError( current_omega );
                  if( t == 1 )
                     mu_last_candidate = mu_last_restart;
               }
#ifdef PRINTING
               std::cout << "Restarting errs.: last restart = " << mu_last_restart << std::endl;
               std::cout << "Sufficient test: " << mu_candidate << " < " << beta_sufficient << " * " << mu_last_restart << " = "
                         << beta_sufficient * mu_last_restart << std::endl;
               /*printf( "Sufficient test: %g < %g * %g = %g\n",
                       mu_candidate,
                       beta_sufficient,
                       mu_last_restart,
                       beta_sufficient * mu_last_restart );*/
#endif

               if( mu_candidate <= beta_sufficient * mu_last_restart ) {
                  std::cout << "SUFFICIENT restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;
                  mu_last_restart = mu_candidate;
                  kkt_last_restart = kkt_candidate;
                  if( writeConvergenceGraphs )
                     restarts_file << k << " SUFFICIENT " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" ) << std::endl;
                  break;
               }

#ifdef PRINTING
               std::cout << "Necessary test: " << mu_candidate << " < " << beta_necessary << " * " << mu_last_restart << " = "
                         << beta_necessary * mu_last_restart << " and " << mu_candidate << " > " << mu_last_candidate
                         << std::endl;
               /*printf( "Necessary test k = %d t = %d: %g < %g * %g = %g and %g > %g\n",
                       k,
                       t,
                       mu_candidate,
                       beta_necessary,
                       mu_last_restart,
                       beta_necessary * mu_last_restart,
                       mu_candidate,
                       mu_last_candidate );*/
#endif

               if( mu_candidate <= beta_necessary * mu_last_restart && mu_candidate > mu_last_candidate ) {
                  std::cout << "NECESSARY restart to " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" )
                            << " at k = " << k << " t = " << t << std::endl;

                  mu_last_restart = mu_candidate;
                  kkt_last_restart = kkt_candidate;
                  if( writeConvergenceGraphs )
                     restarts_file << k << " NECESSARY " << ( mu_averaged <= mu_current ? "AVERAGE" : "CURRENT" ) << std::endl;
                  break;
               }
            }  // if( restarting != PDLPRestarting::None )
            else
               z_candidate = z_averaged;
            z_last_iteration = z_current;
            mu_last_candidate = mu_candidate;
         }  // while( t < max_restarting_steps && k < max_iterations );
      }  // if( this->averaging )
      auto new_x_view = z_candidate.getView( 0, n );
      auto new_y_view = z_candidate.getView( n, n + m1 + m2 );

      auto [ primal_feasibility, dual_feasibility, primal_objective, dual_objective ] = kkt_candidate;

      const RealType epsilon = 1.0e-4;
      const RealType relative_duality_gap = kkt_candidate.getRelativeDualityGap();
      const RealType relative_primal_feasibility = primal_feasibility / ( 1 + l2Norm( q ) );
      const RealType relative_dual_feasibility = dual_feasibility / ( 1 + l2Norm( c ) );

#ifdef PRINTING
      //std::cout << "primal feas. " << primal_feasibility << " dual feas. " << dual_feasibility << " primal obj. "
      //          << primal_objective << " dual obj. " << dual_objective << " duality gap " << duality_gap << std::endl;
      std::cout << "Termination check: " << primal_feasibility << "|" << epsilon * ( 1 + l2Norm( q ) ) << " "
                << dual_feasibility << "|" << epsilon * ( 1 + l2Norm( c ) ) << " " << relative_duality_gap << "|" << epsilon
                << " : primal obj. " << primal_objective << " dual obj. " << dual_objective << std::endl;
#endif

      if( relative_duality_gap < epsilon && relative_primal_feasibility < epsilon && relative_dual_feasibility < epsilon ) {
         solverTimer.stop();
         std::cout << "===============================" << std::endl;
         std::cout << "SOLUTION FOUND" << std::endl;
         std::cout << "PRIMAL OBJECTIVE: " << kkt_candidate.getPrimalObjective() << std::endl;
         std::cout << "DUAL OBJECTIVE: " << kkt_candidate.getDualObjective() << std::endl;
         std::cout << "DUALITY GAP: " << kkt_candidate.getDualityGap() << " / " << relative_duality_gap << std::endl;
         std::cout << "PRIMAL FEASIBILITY: " << kkt_candidate.getPrimalFeasibility() << " / " << relative_primal_feasibility
                   << std::endl;
         std::cout << "DUAL FEASIBILITY: " << kkt_candidate.getDualFeasibility() << " / " << relative_dual_feasibility
                   << std::endl;
         std::cout << "ITERATIONS: " << k << std::endl;
         std::cout << "#Kx COMPUTATIONS:" << this->KxComputations << std::endl;
         std::cout << "#KTy COMPUTATIONS:" << this->KTyComputations << std::endl;
         std::cout << "SOLVER TIME: " << solverTimer.getRealTime() << std::endl;
         std::cout << "MATRIX-VECTOR PRODUCTS TIME: " << this->spmvTimer.getRealTime() << std::endl;
         //std::cout << "X: " << new_x_view << std::endl;
         //std::cout << "D2 * X: " << D2 * new_x_view << std::endl;
         //std::cout << "Y: " << new_y_view << std::endl;
         //x = new_x_view;
         //y = new_y_view;
         return { true, dual_objective, kkt_candidate.getRelativeDualityGap() };
      }
      else {
         std::cout << "ITER: " << std::setw( 6 ) << k << " NORMS=(" << std::setw( 10 ) << l2Norm( new_x_view ) << ", "
                   << std::setw( 10 ) << l2Norm( new_y_view ) << ") INV.STEP : " << std::setw( 10 ) << 1.0 / current_eta
                   << " PRIMAL WEIGHT: " << std::setw( 10 ) << current_omega << " OBJECTIVE : ( " << std::setw( 10 )
                   << primal_objective << ", " << std::setw( 12 ) << dual_objective << " )   FEASIBILITY: ( " << std::setw( 12 )
                   << relative_primal_feasibility << ", " << std::setw( 12 ) << relative_dual_feasibility
                   << " )   DUAL.GAP: " << std::setw( 10 ) << kkt_candidate.getDualityGap() << std::endl;
      }

      //Compute new parameter omega
      if( this->adaptivePrimalWeight ) {
         RealType delta_x = l2Norm( new_x_view - x );
         RealType delta_y = l2Norm( new_y_view - y );
         if( delta_x > 1.0e-10 && delta_y > 1.0e-10 ) {
            const RealType theta = 0.5;
            current_omega = exp( theta * log( delta_y / delta_x ) + ( 1.0 - theta ) * log( current_omega ) );
         }
#ifdef PRINTING
         std::cout << "Omega update: primal diff = " << delta_x << " dual diff = " << delta_y
                   << " new omega = " << current_omega << std::endl;
#endif
         x = new_x_view;
         y = new_y_view;
      }
   }
   return { false, 0.0, 0.0 };
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::adaptiveStep( const VectorType& in_z,
                                                 VectorType& out_z,
                                                 const IndexType k,
                                                 RealType& current_omega,
                                                 RealType& current_eta )
{
   auto in_x = in_z.getConstView( 0, n );
   auto in_y = in_z.getConstView( n, N );
   auto out_x = out_z.getView( 0, n );
   auto out_y = out_z.getView( n, N );

   VectorType delta_y( m, 0 ), delta_x( n, 0 ), aux( n, 0 );

   auto KTy_view = KTy.getView();
   computeKTy( in_y, KTy_view );

   auto Kx_view = Kx.getView();
   computeKx( in_x, Kx_view );

   while( true ) {
      const RealType tau = current_eta / current_omega;
      const RealType sigma = current_eta * current_omega;

#ifdef PRINTING
      std::cout << "=====================================================================" << std::endl;
      std::cout << this->adaptive_k - 2 << " eta: " << current_eta << " omega: " << current_omega << " primal step : " << tau
                << " dual step : " << sigma << std::endl;
#endif

      // Primal step
      computePrimalStep( in_x, KTy, tau, out_x );

      // Dual step
      auto Kx_new_view = Kx_new.getView();
      computeKx( out_x, Kx_new_view );
      computeDualStep( in_y, Kx, Kx_new, sigma, out_y );

#ifdef PRINTING
      std::cout << "Adpt. step       x = " << l2Norm( in_x ) << std::endl;
      std::cout << "Adpt. step       y = " << l2Norm( in_y ) << std::endl;
      std::cout << "Adpt. step    AT_y = " << l2Norm( KT_y ) << std::endl;
      std::cout << "Adpt. step   out_x = " << l2Norm( out_x ) << std::endl;
      std::cout << "Adpt. step   out_y = " << l2Norm( out_y ) << std::endl;
#endif

      // Compute new parameter eta
      delta_x = out_x - in_x;
      delta_y = out_y - in_y;
      const RealType movement = 0.5 * ( current_omega * ( delta_x, delta_x ) + ( delta_y, delta_y ) / current_omega );

      const RealType interaction = abs( dot( Kx_new - Kx, delta_y ) );
      const RealType max_eta = interaction > 0 ? movement / interaction : std::numeric_limits< RealType >::infinity();
      RealType new_eta;
      if( this->adaptive_k == 0 && max_eta == std::numeric_limits< RealType >::infinity() )
         new_eta = ( 1.0 + pow( this->adaptive_k + 1, -0.6 ) ) * current_eta;
      else
         new_eta = min( ( 1.0 - pow( this->adaptive_k + 1, -0.3 ) ) * max_eta,
                        ( 1.0 + pow( this->adaptive_k + 1, -0.6 ) ) * current_eta );
      TNL_ASSERT_GT( new_eta, 0, "new_eta <= 0" );
      //std::cout << "   Adaptive step: k = " << this->adaptive_k << " new eta = " << new_eta << std::endl;

#ifdef PRINTING
      std::cout << "   Movement: dX " << ( delta_x, delta_x ) << " dY " << ( delta_y, delta_y ) << std::endl;
      std::cout << "   delta_x = " << l2Norm( delta_x ) << "\n   delta_y = " << l2Norm( delta_y ) << std::endl;
      std::cout << "k: " << this->adaptive_k << " movement : " << movement << " interaction : " << interaction
                << " step limit : " << max_eta << " new eta: " << new_eta << std::endl;
#endif
      this->adaptive_k++;
      if( current_eta < max_eta ) {
         current_eta = new_eta;
#ifdef PRINTING
         std::cout << "End of adaptive step." << std::endl;
         std::cout << "Adpt. step   out_x = " << l2Norm( out_x ) << std::endl;
         std::cout << "Adpt. step   out_y = " << l2Norm( out_y ) << std::endl;
#endif
         Kx = Kx_new;
         return;
      }
      else
         current_eta = new_eta;
   }
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::computeKx( const ConstVectorView& x, VectorView& Kx )
{
   spmvTimer.start();
   K.vectorProduct( x, Kx, segmentsReductionKernel );
   spmvTimer.stop();
   this->KxComputations++;
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::computeKTy( const ConstVectorView& y, VectorView& KTy )
{
   spmvTimer.start();
   KT.vectorProduct( y, KTy, segmentsReductionKernel );
   spmvTimer.stop();
   this->KTyComputations++;
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::computePrimalStep( const ConstVectorView& x,
                                                      const VectorView& KTy,
                                                      const RealType& tau,
                                                      VectorView& x_new )
{
   x_new = minimum( u, maximum( l, x - tau * ( c - KTy ) ) );
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::computeDualStep( const ConstVectorView& y,
                                                    const VectorView& Kx,
                                                    const VectorView& Kx_new,
                                                    const RealType& sigma,
                                                    VectorView& y_new )
{
   y_new = y + sigma * ( q - 2 * Kx_new + Kx );
   if( this->inequalitiesFirst ) {
      if( m1 > 0 )
         y_new.getView( 0, m1 ) = maximum( 0, y_new.getView( 0, m1 ) );
   }
   else {
      if( m2 > 0 )
         y_new.getView( m1, m ) = maximum( 0, y_new.getView( m1, m ) );
   }
}

template< typename LPProblem_, typename SolverMonitor >
void
PDLP< LPProblem_, SolverMonitor >::computeLambda( const VectorType& c,
                                                  const VectorType& KTy,
                                                  const VectorType& l,
                                                  const VectorType& u,
                                                  VectorType& lambda )
{
   auto c_view = c.getConstView();
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
#ifdef PRINTING
   std::cout << "LAMBDA:      y = " << l2Norm( y )                    //
             << "\nLAMBDA:c - KTy = " << l2Norm( c_view - KTy_view )  //
             << "\nLAMBDA: lambda = " << l2Norm( lambda )             //
             << std::endl;
#endif
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::computePrimalFeasibility( const VectorType& q, const VectorType& Kx ) -> RealType
{
#ifdef PRINTING
   std::cout << "PRIMAL.FS:      x = " << l2Norm( x )  //
             << "\nPRIMAL.FS:     Kx = " << l2Norm( Kx ) << std::endl;
#endif

   // TODO: Rewrite using reduction
   VectorType Kx_( m );
   if( this->inequalitiesFirst ) {
      if( m1 > 0 )
         Kx_.getView( 0, m1 ) = maximum( q.getConstView( 0, m1 ) - Kx.getConstView( 0, m1 ), 0 );  // ( h - Gx)^+
      if( m > m1 )
         Kx_.getView( m1, m ) = Kx.getConstView( m1, m ) - q.getConstView( m1, m );  // ( Ax - b )
   }
   else {
      if( m1 > 0 )
         Kx_.getView( 0, m1 ) = Kx.getConstView( 0, m1 ) - q.getConstView( 0, m1 );  //( Ax - b)
      if( m > m1 )
         Kx_.getView( m1, m ) = maximum( q.getConstView( m1, m ) - Kx.getConstView( m1, m ), 0 );  // ( h - Gx)^+
   }
   return l2Norm( Kx_ );
}

template< typename LPProblem_, typename SolverMonitor >
auto
PDLP< LPProblem_, SolverMonitor >::primalDualGap( const VectorView& z, const VectorView& z_ref ) -> RealType
{
   auto x_view = z.getConstView( 0, n );
   auto y_view = z.getConstView( n, N );

   VectorType g( N ), g_l( N ), g_u( N );
   auto g_1 = g.getView( 0, n );
   auto g_2 = g.getView( n, N );

   computeKTy( y_view, g_1 );  // g_1 = KT * y_view
   g_1 = c - g_1;
   computeKx( x_view, g_2 );  // g_2 = K * x_view
   g_2 = g_2 - q;

   g_l.getView( 0, n ) = l;
   g_u.getView( 0, n ) = u;

   if( this->inequalitiesFirst ) {
      if( m1 > 0 )
         g_l.getView( n, n + m1 ) = 0.0;
      if( m2 > 0 )
         g_l.getView( n + m1, N ) = -std::numeric_limits< RealType >::infinity();
   }
   else {
      if( m1 > 0 )
         g_l.getView( n, n + m1 ) = -std::numeric_limits< RealType >::infinity();
      if( m2 > 0 )
         g_l.getView( n + m1, N ) = 0.0;
   }
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
PDLP< LPProblem_, SolverMonitor >::KKT( const VectorView& z, const VectorType& Kx, const VectorType& KTy ) -> KKTDataType
{
   auto x = z.getConstView( 0, n );
   auto y = z.getConstView( n, N );
   auto c_view = c.getConstView();

   // Compute error of the primal feasibility
   const RealType primal_feasibility = computePrimalFeasibility( q, Kx );

   // Compute error of the dual feasibility
   computeLambda( c, KTy, l, u, lambda );
   const RealType dual_feasibility = l2Norm( c - KTy - lambda );

   // Compute the primal objective
   const RealType primal_objective = ( c, x );

   // Compute the dual objective
   auto lambda_view = lambda.getConstView();
   const RealType dual_objective = ( q, y ) + ( filtered_l, maximum( lambda_view, 0 ) )
                                 + ( filtered_u, minimum( lambda_view, 0 ) );  // cuPDLP-C has + here, original paper has -
   //std::cout << "( q, y ) = " << ( q, y ) << std::endl;
   //std::cout << "lower filter = " << ( filtered_l, maximum( lambda_view, 0 ) ) << std::endl;
   //std::cout << "upper filter = " << ( filtered_u, minimum( lambda_view, 0 ) ) << std::endl;

   return { primal_feasibility, dual_feasibility, primal_objective, dual_objective };
}

template< typename Real >
Real
KKTData< Real >::getKKTError( const Real& omega ) const
{
   const Real omega_sqr = omega * omega;
   const Real error =
      sqrt( omega_sqr * primal_feasibility * primal_feasibility + 1.0 / omega_sqr * ( dual_feasibility * dual_feasibility )
            + pow( primal_objective - dual_objective, 2 ) );

#ifdef PRINTING
   std::cout << " omega sqr. = " << omega_sqr << " primal feas. = " << primal_feasibility
             << " dual feas. = " << dual_feasibility << " primal obj. = " << primal_objective
             << " dual obj. = " << dual_objective << " error = " << error << std::endl;
#endif
   return error;
}

template< typename Real >
Real
KKTData< Real >::getRelativeDualityGap() const
{
   return std::abs( ( primal_objective - dual_objective ) / ( 1 + std::abs( primal_objective ) + std::abs( dual_objective ) ) );
}

}  // namespace TNL::Solvers::Optimization
