// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/LinearCombination.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/detail/ODESolverEvaluator.h>

#ifndef DOXYGEN_ONLY
namespace TNL::Solvers::ODE {

// Specialization for static vectors

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
ODESolver< Method, Value, SolverMonitor, true >::ODESolver()
{
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
}

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
ODESolver< Method, Value, SolverMonitor, true >::ODESolver( const ODESolver& solver )
: StaticExplicitSolver< GetValueType_t< Value >, std::size_t >( solver )
{
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
   this->method = solver.method;
}

template< typename Method, typename Value, typename SolverMonitor >
void
ODESolver< Method, Value, SolverMonitor, true >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   ExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "integrator-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
}

template< typename Method, typename Value, typename SolverMonitor >
bool
ODESolver< Method, Value, SolverMonitor, true >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< Value, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "integrator-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "integrator-adaptivity" ) );
   return true;
}

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
Method&
ODESolver< Method, Value, SolverMonitor, true >::getMethod()
{
   return this->method;
}

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
const Method&
ODESolver< Method, Value, SolverMonitor, true >::getMethod() const
{
   return this->method;
}

template< typename Method, typename Value, typename SolverMonitor >
template< typename RHSFunction, typename... Params >
__cuda_callable__
bool
ODESolver< Method, Value, SolverMonitor, true >::solve( VectorType& u, RHSFunction&& rhsFunction, Params&&... params )
{
   TNL_ASSERT_GT( this->getTau(), 0.0, "The time step for the ODE solver is zero." );

   this->init( u );

   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   // Start the main loop
   while( this->checkNextIteration() ) {
      this->iterate( u, time, currentTau, rhsFunction, params... );
      if( ! this->nextIteration() )
         return false;

      // Tune the new time step.
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
void
ODESolver< Method, Value, SolverMonitor, true >::init( const VectorType& u )
{
   for( int i = 0; i < Stages; i++ )
      k_vectors[ i ] = 0;
   kAux = 0;
}

template< typename Method, typename Value, typename SolverMonitor >
template< typename RHSFunction, typename... Params >
__cuda_callable__
void
ODESolver< Method, Value, SolverMonitor, true >::iterate( VectorType& u,
                                                          RealType& time,
                                                          RealType& currentTau,
                                                          RHSFunction&& rhsFunction,
                                                          Params&&... params )
{
   using ErrorCoefficients = detail::ErrorCoefficientsProxy< Method >;
   using UpdateCoefficients = detail::UpdateCoefficientsProxy< Method >;
   using Containers::linearCombination;

   RealType error( 0.0 );
   bool compute( true );
   while( compute ) {
      detail::StaticODESolverEvaluator< Method >::computeKVectors(
         k_vectors, time, currentTau, u, kAux, rhsFunction, params... );
      if constexpr( Method::isAdaptive() )
         if( this->adaptivity )
            error = currentTau * max( abs( linearCombination< ErrorCoefficients >( k_vectors ) ) );

      if( this->adaptivity == 0.0 || error < this->adaptivity ) {
         RealType lastResidue = this->getResidue();

         this->setResidue(
            addAndReduceAbs( u, currentTau * linearCombination< UpdateCoefficients >( k_vectors ), TNL::Plus{}, 0.0 )
            / currentTau );
         time += currentTau;

         // When time is close to stopTime the new residue may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );
         compute = false;
      }

      // Compute the new time step.
      RealType newTau = currentTau;
      if( this->adaptivity != 0.0 && error != 0.0 )
         newTau = currentTau * 0.8 * TNL::pow( this->adaptivity / error, 0.2 );
      currentTau = min( newTau, this->getMaxTau() );
   }
}

template< typename Method, typename Value, typename SolverMonitor >
__cuda_callable__
void
ODESolver< Method, Value, SolverMonitor, true >::reset()
{}

// Specialization for dynamic vectors
template< typename Method, typename Vector, typename SolverMonitor >
ODESolver< Method, Vector, SolverMonitor, false >::ODESolver()
{
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
}

template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor, false >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   ExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "integrator-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
}

template< typename Method, typename Vector, typename SolverMonitor >
bool
ODESolver< Method, Vector, SolverMonitor, false >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< Vector, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "integrator-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "integrator-adaptivity" ) );
   return true;
}

template< typename Method, typename Vector, typename SolverMonitor >
Method&
ODESolver< Method, Vector, SolverMonitor, false >::getMethod()
{
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
const Method&
ODESolver< Method, Vector, SolverMonitor, false >::getMethod() const
{
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction, typename... Params >
bool
ODESolver< Method, Vector, SolverMonitor, false >::solve( VectorType& u, RHSFunction&& rhsFunction, Params&&... params )
{
   if( this->getTau() == 0.0 ) {
      std::cerr << "The time step for the ODE solver is zero.\n";
      return false;
   }

   this->init( u );

   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   // Start the main loop
   while( this->checkNextIteration() ) {
      this->iterate( u, time, currentTau, rhsFunction, params... );
      if( ! this->nextIteration() )
         return false;

      // Tune the new time step.
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor, false >::init( const VectorType& u )
{
   for( int i = 0; i < Stages; i++ ) {
      k_vectors[ i ].setLike( u );
      k_vectors[ i ] = 0;
   }
   kAux.setLike( u );
   kAux = 0;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction, typename... Params >
void
ODESolver< Method, Vector, SolverMonitor, false >::iterate( VectorType& u,
                                                            RealType& time,
                                                            RealType& currentTau,
                                                            RHSFunction&& rhsFunction,
                                                            Params&&... params )
{
   using ErrorCoefficients = detail::ErrorCoefficientsProxy< Method >;
   using UpdateCoefficients = detail::UpdateCoefficientsProxy< Method >;
   using Containers::linearCombination;

   using VectorView = typename Vector::ViewType;
   std::array< VectorView, Stages > k_views;

   // Setup the supporting vectors views which will be passed to rhsFunction
   for( int i = 0; i < Stages; i++ ) {
      k_views[ i ].bind( k_vectors[ i ] );
   }

   RealType error( 0.0 );
   bool compute( true );
   while( compute ) {
      detail::ODESolverEvaluator< Method >::computeKVectors(
         k_views, time, currentTau, u.getView(), kAux.getView(), rhsFunction, params... );

      if constexpr( Method::isAdaptive() )
         if( this->adaptivity )
            error = currentTau * max( abs( linearCombination< ErrorCoefficients >( k_vectors ) ) );

      if( this->adaptivity == 0.0 || error < this->adaptivity ) {
         RealType lastResidue = this->getResidue();

         this->setResidue(
            addAndReduceAbs( u, currentTau * linearCombination< UpdateCoefficients >( k_vectors ), TNL::Plus{}, 0.0 )
            / ( currentTau * (RealType) u.getSize() ) );
         time += currentTau;

         // When time is close to stopTime the new residue may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );
         compute = false;
      }

      // Compute the new time step.
      RealType newTau = currentTau;
      if( this->adaptivity != 0.0 && error != 0.0 )
         newTau = currentTau * 0.8 * TNL::pow( this->adaptivity / error, 0.2 );
      currentTau = min( newTau, this->getMaxTau() );
   }
}

template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor, false >::reset()
{
   for( int i = 0; i < Stages; i++ ) {
      k_vectors[ i ].reset();
   }
   kAux.reset();
}

}  // namespace TNL::Solvers::ODE
#endif  // DOXYGEN_ONLY
