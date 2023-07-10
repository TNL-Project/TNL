// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Expressions/LinearCombination.h>
#include <TNL/Solvers/ODE/StaticODESolver.h>
#include <TNL/Solvers/ODE/detail/ODESolverEvaluator.h>

namespace TNL::Solvers::ODE {

template< typename Method, typename Vector >
void
StaticODESolver< Method, Vector >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Method, typename Vector >
bool
StaticODESolver< Method, Vector >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   StaticExplicitSolver< Vector >::setup( parameters, prefix );
   return true;
}

template< typename Method, typename Vector >
__cuda_callable__ Method&
StaticODESolver< Method, Vector >::getMethod() {
   return this->method;
}

template< typename Method, typename Vector >
__cuda_callable__ const Method&
StaticODESolver< Method, Vector >::getMethod() const {
   return this->method;
}

template< typename Method, typename Vector >
template< typename RHSFunction >
__cuda_callable__ bool
StaticODESolver< Method, Vector >::solve( VectorType& u, RHSFunction&& rhsFunction )
{
   if( this->getTau() == 0.0 ) {
      return false;
   }
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

      detail::ODESolverEvaluator< Method >::computeKVectors( k_views, time, currentTau, u.getConstView(), kAux.getView(), rhsFunction );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if constexpr( Method::isAdaptive() )
         error = Method::getError( k_views, currentTau );

      VectorType update( u );
      if( method.acceptStep( error ) ) {
         RealType lastResidue = this->getResidue();

         using UpdateCoefficients = UpdateCoefficientsExtractor< Method >;
         using UpdateExpression = Containers::Expressions::LinearCombination< UpdateCoefficients, Vector >;
         const VectorType update = UpdateExpression::evaluateArray( k_vectors );
         u += currentTau * update;
         this->setResidue( abs( update ) );
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

      //std::cout << "time = " << time << ", tau = " << currentTau << ", residue = " << this->getResidue() << " u = " << u << std::endl;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

}  // namespace TNL::Solvers::ODE
