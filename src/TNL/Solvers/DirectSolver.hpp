// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include "DirectSolver.h"

namespace TNL::Solvers {

template< typename Real, typename Index, typename SolverMonitor >
constexpr bool
DirectSolver< Real, Index, SolverMonitor >::isIterativeSolver()
{
   return false;
}

template< typename Real, typename Index, typename SolverMonitor >
constexpr bool
DirectSolver< Real, Index, SolverMonitor >::isDirectSolver()
{
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
void
DirectSolver< Real, Index, SolverMonitor >::configSetup( Config::ConfigDescription& config, const std::string& prefix )
{
   if( config.getEntry( prefix + "refresh-rate" ) == nullptr )
      config.addEntry< int >( prefix + "refresh-rate", "Number of milliseconds between solver monitor refreshes.", 500 );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
DirectSolver< Real, Index, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const std::string& prefix )
{
   if( parameters.checkParameter( prefix + "refresh-rate" ) )
      this->setRefreshRate( parameters.getParameter< int >( prefix + "refresh-rate" ) );
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
void
DirectSolver< Real, Index, SolverMonitor >::setResidue( const Real& residue )
{
   this->currentResidue = residue;
   if( this->solverMonitor )
      this->solverMonitor->setResidue( this->getResidue() );
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
DirectSolver< Real, Index, SolverMonitor >::getResidue() const
{
   return this->currentResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
void
DirectSolver< Real, Index, SolverMonitor >::setRefreshRate( const Index& refreshRate )
{
   this->refreshRate = refreshRate;
   if( this->solverMonitor )
      this->solverMonitor->setRefreshRate( this->refreshRate );
}

template< typename Real, typename Index, typename SolverMonitor >
void
DirectSolver< Real, Index, SolverMonitor >::setSolverMonitor( SolverMonitorType& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
   this->solverMonitor->setRefreshRate( this->refreshRate );
}

}  // namespace TNL::Solvers
