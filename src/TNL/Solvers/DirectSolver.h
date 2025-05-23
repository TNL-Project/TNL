// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <string>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/DirectSolverMonitor.h>

namespace TNL::Solvers {

/**
 * \brief Base class for direct solvers.
 *
 * \tparam Real is a floating point type used for computations.
 * \tparam Index is an indexing type.
 * \tparam DirectSolverMonitor< Real, Index > is type of an object used for monitoring of the convergence.
 */
template< typename Real, typename Index, typename SolverMonitor = DirectSolverMonitor< double, int > >
class DirectSolver
{
public:
   /**
    * \brief Floating point type used for computations.
    */
   using RealType = Real;

   /**
    * \brief Indexing type.
    */
   using IndexType = Index;

   /**
    * \brief Type of an object used for monitoring of the convergence.
    */
   using SolverMonitorType = SolverMonitor;

   /**
    * \brief Returns \e true if the solver is an iterative solver.
    */
   static constexpr bool
   isIterativeSolver();

   /**
    * \brief Returns \e true if the solver is a direct solver.
    */
   static constexpr bool
   isDirectSolver();

   /**
    * \brief Default constructor.
    */
   DirectSolver() = default;

   /**
    * \brief This method defines configuration entries for setup of the iterative solver.
    *
    * The following entries are defined:
    *
    * \e refresh-rate - number of milliseconds between solver monitor refreshes.
    *
    * \param config contains description of configuration parameters.
    * \param prefix is a prefix of particular configuration entries.
    */
   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   /**
    * \brief Method for setup of the direct solver based on configuration parameters.
    *
    * \param parameters contains values of the define configuration entries.
    * \param prefix is a prefix of particular configuration entries.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   /**
    * \brief Sets the residue reached at the current iteration.
    *
    * \param residue reached at the current iteration.
    */
   void
   setResidue( const Real& residue );

   /**
    * \brief Gets the residue reached at the current iteration.
    *
    * \return residue reached at the current iteration.
    */
   const Real&
   getResidue() const;

   /**
    * \brief Sets the refresh rate (in milliseconds) for the solver monitor.
    *
    * \param refreshRate of the solver monitor in milliseconds.
    */
   void
   setRefreshRate( const Index& refreshRate );

   /**
    * \brief Sets the solver monitor object.
    *
    * The solver monitor is an object for monitoring the status of the iterative solver.
    * Usually it prints the number of iterations, current residue or elapsed time.
    *
    * \param solverMonitor is an object for monitoring the iterative solver.
    */
   void
   setSolverMonitor( SolverMonitorType& solverMonitor );

protected:
   SolverMonitor* solverMonitor = nullptr;

   Index refreshRate = 1;

   Real currentResidue = std::numeric_limits< RealType >::max();
};

}  // namespace TNL::Solvers

#include "DirectSolver.hpp"
