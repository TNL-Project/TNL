// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Expressions/LinearCombination.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/detail/ODESolverEvaluator.h>

namespace TNL::Solvers::ODE {

////
// Specialization for static vectors and numbers
template< typename Method, typename Vector, typename SolverMonitor >
__cuda_callable__
ODESolver< Method, Vector, SolverMonitor, true >::ODESolver() {
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
}

template< typename Method, typename Vector, typename SolverMonitor >
__cuda_callable__
ODESolver< Method, Vector, SolverMonitor, true >::
ODESolver( const ODESolver& solver )
: StaticExplicitSolver< typename GetRealType< Vector >::type, typename GetIndexType < Vector >::type >( solver )
{
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
   this->method = solver.method;
}


template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor, true >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   ExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "integrator-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
}

template< typename Method, typename Vector, typename SolverMonitor >
bool
ODESolver< Method, Vector, SolverMonitor, true >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< Vector, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "integrator-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "integrator-adaptivity" ) );
   return true;
}

template< typename Method, typename Vector, typename SolverMonitor >
__cuda_callable__
Method&
ODESolver< Method, Vector, SolverMonitor, true >::getMethod() {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
__cuda_callable__
const Method&
ODESolver< Method, Vector, SolverMonitor, true >::getMethod() const {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction >
__cuda_callable__
bool
ODESolver< Method, Vector, SolverMonitor, true >::solve( VectorType& u, RHSFunction&& rhsFunction )
{
   using ErrorCoefficients = detail::ErrorCoefficientsExtractor< Method >;
   using ErrorExpression = Containers::Expressions::LinearCombination< ErrorCoefficients, Vector >;
   using UpdateCoefficients = detail::UpdateCoefficientsExtractor< Method >;
   using UpdateExpression = Containers::Expressions::LinearCombination< UpdateCoefficients, Vector >;


   if( this->getTau() == 0.0 ) {
      std::cerr << "The time step for the ODE solver is zero." << std::endl;
      return false;
   }

   for( int i = 0; i < Stages; i++ )
       k_vectors[ i ] = 0;

   /////
   // Setup the supporting vectors
   kAux = 0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( this->checkNextIteration() ) {

      detail::ODESolverEvaluator< Method >::computeKVectors( k_vectors, time, currentTau, u, kAux, rhsFunction );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if constexpr( Method::isAdaptive() )
         if( this->adaptivity )
            error = currentTau * max( abs( ErrorExpression::evaluateArray( k_vectors ) ) );

      if( this->adaptivity == 0.0 || error < this->adaptivity ) {
         RealType lastResidue = this->getResidue();

         if constexpr( std::is_arithmetic_v< ValueType > ) {
            ValueType update = UpdateExpression::evaluateArray( k_vectors );
            u += currentTau * update;
            this->setResidue( abs( update ) );
         }
         else {
            this->setResidue(
               addAndReduceAbs( u, currentTau * UpdateExpression::evaluateArray( k_vectors ), TNL::Plus{}, 0.0 ) / currentTau );
         }
         time += currentTau;

         /////
         // When time is close to stopTime the new residue may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      RealType newTau = currentTau;
      if( adaptivity != 0.0 && error != 0.0 )
         newTau = currentTau * 0.8 * TNL::pow( adaptivity / error, 0.2 );
      currentTau = min( newTau, this->getMaxTau() );
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

////
// Specialization for dynamic vectors
template< typename Method, typename Vector, typename SolverMonitor >
ODESolver< Method, Vector, SolverMonitor, false >::ODESolver() {
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
ODESolver< Method, Vector, SolverMonitor, false >::getMethod() {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
const Method&
ODESolver< Method, Vector, SolverMonitor, false >::getMethod() const {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction >
bool
ODESolver< Method, Vector, SolverMonitor, false >::solve( VectorType& u, RHSFunction&& rhsFunction )
{
   using ErrorCoefficients = detail::ErrorCoefficientsExtractor< Method >;
   using ErrorExpression = Containers::Expressions::LinearCombination< ErrorCoefficients, Vector >;
   using UpdateCoefficients = detail::UpdateCoefficientsExtractor< Method >;
   using UpdateExpression = Containers::Expressions::LinearCombination< UpdateCoefficients, Vector >;


   if( this->getTau() == 0.0 ) {
      std::cerr << "The time step for the ODE solver is zero." << std::endl;
      return false;
   }

   using VectorView = typename Vector::ViewType;
   std::array< VectorView, Stages > k_views;

   /////
   // Setup the supporting vectors
   for( int i = 0; i < Stages; i++ ) {
       k_vectors[ i ].setLike( u );
       k_vectors[ i ] = 0;
       k_views[ i ].bind( k_vectors[ i ] );
   }
   kAux.setLike( u );
   kAux = 0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( this->checkNextIteration() ) {

      detail::ODESolverEvaluator< Method >::computeKVectors( k_views, time, currentTau, u.getView(), kAux.getView(), rhsFunction );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if constexpr( Method::isAdaptive() )
         if( this->adaptivity )
            error = currentTau * max( abs( ErrorExpression::evaluateArray( k_vectors ) ) );

      if( this->adaptivity == 0.0 || error < this->adaptivity ) {
         RealType lastResidue = this->getResidue();

         this->setResidue(
            addAndReduceAbs( u, currentTau * UpdateExpression::evaluateArray( k_vectors ), TNL::Plus{}, 0.0 ) /
            ( currentTau * (RealType) u.getSize() ) );
         time += currentTau;

         /////
         // When time is close to stopTime the new residue may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      RealType newTau = currentTau;
      if( adaptivity != 0.0 && error != 0.0 )
         newTau = currentTau * 0.8 * TNL::pow( adaptivity / error, 0.2 );
      currentTau = min( newTau, this->getMaxTau() );
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

}  // namespace TNL::Solvers::ODE
