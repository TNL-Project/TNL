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

template< typename Method, typename Vector, typename SolverMonitor >
ODESolver< Method, Vector, SolverMonitor >::ODESolver() {
   // It is better to turn off the convergence check for the ODE solver by default.
   this->setConvergenceResidue( 0.0 );
}

template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   ExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Method, typename Vector, typename SolverMonitor >
bool
ODESolver< Method, Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< Vector, SolverMonitor >::setup( parameters, prefix );
   return true;
}

template< typename Method, typename Vector, typename SolverMonitor >
Method&
ODESolver< Method, Vector, SolverMonitor >::getMethod() {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
const Method&
ODESolver< Method, Vector, SolverMonitor >::getMethod() const {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction >
bool
ODESolver< Method, Vector, SolverMonitor >::solve( VectorType& u, RHSFunction&& rhsFunction )
{
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
         if( this->adaptivity ) {
            using ErrorCoefficients = detail::ErrorCoefficientsExtractor< Method >;
            using ErrorExpression = Containers::Expressions::LinearCombination< ErrorCoefficients, Vector >;
            error = currentTau * max( abs( ErrorExpression::evaluateArray( k_vectors ) ) );
      }

      if( this->adaptivity == 0.0 || error < this->adaptivity ) {
         RealType lastResidue = this->getResidue();

         using UpdateCoefficients = detail::UpdateCoefficientsExtractor< Method >;
         using UpdateExpression = Containers::Expressions::LinearCombination< UpdateCoefficients, Vector >;
         this->setResidue(
            addAndReduceAbs( u, currentTau * UpdateExpression::evaluateArray( k_vectors ), TNL::Plus{}, 0.0 ) /
            ( currentTau * (RealType) u.getSize() ) );
         time += currentTau;

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      currentTau = min( method.computeTau( error, currentTau ), this->getMaxTau() );
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
