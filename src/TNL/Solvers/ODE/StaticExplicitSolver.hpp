// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/ODE/StaticExplicitSolver.h>

namespace TNL::Solvers::ODE {

template< typename Real, typename Index >
void
StaticExplicitSolver< Real, Index >::configSetup( Config::ConfigDescription& config, const std::string& prefix )
{
   StaticIterativeSolver< Real, Index >::configSetup( config, prefix );
   StaticIterativeSolver< Real, Index >::configSetup( config, prefix );
   config.addEntry< bool >(
      prefix + "stop-on-steady-state", "The computation stops when steady-state solution is reached.", false );
}

template< typename Real, typename Index >
bool
StaticExplicitSolver< Real, Index >::setup( const Config::ParameterContainer& parameters, const std::string& prefix )
{
   this->stopOnSteadyState = parameters.getParameter< bool >( "stop-on-steady-state" );
   return StaticIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Real, typename Index >
__cuda_callable__
void
StaticExplicitSolver< Real, Index >::setTime( const RealType& time )
{
   this->time = time;
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
StaticExplicitSolver< Real, Index >::getTime() const
{
   return this->time;
}

template< typename Real, typename Index >
__cuda_callable__
void
StaticExplicitSolver< Real, Index >::setTau( const RealType& tau )
{
   this->tau = tau;
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
StaticExplicitSolver< Real, Index >::getTau() const
{
   return this->tau;
}

template< typename Real, typename Index >
__cuda_callable__
void
StaticExplicitSolver< Real, Index >::setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
StaticExplicitSolver< Real, Index >::getMaxTau() const
{
   return this->maxTau;
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
StaticExplicitSolver< Real, Index >::getStopTime() const
{
   return this->stopTime;
}

template< typename Real, typename Index >
__cuda_callable__
void
StaticExplicitSolver< Real, Index >::setStopTime( const RealType& stopTime )
{
   this->stopTime = stopTime;
}

template< typename Real, typename Index >
__cuda_callable__
bool
StaticExplicitSolver< Real, Index >::nextIteration()
{
   // the base class must be used first because it calls checkNextIteration() which checks the current time
   bool result = StaticIterativeSolver< RealType, IndexType >::nextIteration();
   this->setTime( this->getTime() + this->getTau() );
   return result;
}

template< typename Real, typename Index >
__cuda_callable__
bool
StaticExplicitSolver< Real, Index >::checkNextIteration() const
{
   if( this->getTime() >= this->getStopTime() )
      return false;
   if( this->stopOnSteadyState )
      // the base class checks the residue and the number of iterations
      return StaticIterativeSolver< RealType, IndexType >::checkNextIteration();
   return true;
}

template< typename Real, typename Index >
__cuda_callable__
bool
StaticExplicitSolver< Real, Index >::checkConvergence() const
{
   if( this->getTime() >= this->getStopTime() )
      return true;
   if( this->stopOnSteadyState )
      if( StaticIterativeSolver< RealType, IndexType >::checkConvergence() )
         return true;
   // std::cerr << "\nThe solver has not reached the stop time.\n";
   return false;
}

template< typename Real, typename Index >
__cuda_callable__
void
StaticExplicitSolver< Real, Index >::setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

}  // namespace TNL::Solvers::ODE
