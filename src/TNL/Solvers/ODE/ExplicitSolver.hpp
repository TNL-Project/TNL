// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/ODE/ExplicitSolver.h>

namespace TNL::Solvers::ODE {

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::configSetup( Config::ConfigDescription& config, const std::string& prefix )
{
   IterativeSolver< Real, Index >::configSetup( config, prefix );
   config.addEntry< bool >(
      prefix + "stop-on-steady-state", "The computation stops when steady-state solution is reached.", false );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const std::string& prefix )
{
   this->stopOnSteadyState = parameters.getParameter< bool >( "stop-on-steady-state" );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTime( const RealType& time )
{
   this->time = time;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getTime() const
{
   return this->time;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTau( const RealType& tau )
{
   this->tau = tau;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getTau() const
{
   return this->tau;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getMaxTau() const
{
   return this->maxTau;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getStopTime() const
{
   return this->stopTime;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setStopTime( const RealType& stopTime )
{
   this->stopTime = stopTime;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::refreshSolverMonitor( bool force )
{
   if( this->solverMonitor ) {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setTimeStep( this->getTau() );
      this->solverMonitor->setTime( this->getTime() );
      this->solverMonitor->setRefreshRate( this->refreshRate );
   }
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::nextIteration()
{
   // the base class must be used first because it calls checkNextIteration() which checks the current time
   bool result = IterativeSolver< RealType, IndexType, SolverMonitor >::nextIteration();
   this->setTime( this->getTime() + this->getTau() );
   if( this->solverMonitor ) {
      this->solverMonitor->setTime( this->getTime() );
   }
   return result;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::checkNextIteration() const
{
   if( this->getTime() >= this->getStopTime() )
      return false;
   if( this->stopOnSteadyState )
      // the base class checks the residue and the number of iterations
      return IterativeSolver< RealType, IndexType, SolverMonitor >::checkNextIteration();
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::checkConvergence() const
{
   if( this->getTime() >= this->getStopTime() )
      return true;
   if( this->stopOnSteadyState )
      if( IterativeSolver< RealType, IndexType, SolverMonitor >::checkConvergence() )
         return true;
   std::cerr << "\nThe solver has not reached the stop time.\n";
   return false;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

}  // namespace TNL::Solvers::ODE
