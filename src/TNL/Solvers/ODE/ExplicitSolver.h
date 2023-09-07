// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Solvers::ODE {

/**
 * \brief Base class for ODE solvers and explicit solvers od PDEs.
 *
 * See also: \ref TNL::Solvers::ODE::Euler, \ref TNL::Solvers::ODE::Merson.
 *
 * \tparam Real is type of the floating-point arithmetics.
 * \tparam Index is type for indexing.
 * \tparam IterativeSolverMonitor< Real, Index > is
 */
template< typename Real = double, typename Index = int, typename SolverMonitor = IterativeSolverMonitor< Real, Index > >
class ExplicitSolver : public IterativeSolver< Real, Index, SolverMonitor >
{
public:
   /**
    * \brief Type of the floating-point arithmetics.
    */
   using RealType = Real;

   /**
    * \brief Indexing type.
    */
   using IndexType = Index;

   /**
    * \brief Type of the monitor of the convergence of the solver.
    */
   using SolverMonitorType = SolverMonitor;

   /**
    * \brief Default constructor.
    */
   ExplicitSolver() = default;

   /**
    * \brief This method defines configuration entries for setup of the iterative solver.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   /**
    * \brief Method for setup of the iterative solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed sucessfuly.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   /**
    * \brief Settter of the current time of the evolution computed by the solver.
    */
   void
   setTime( const RealType& t );

   /**
    * \brief Getter of the current time of the evolution computed by the solver.
    */
   [[nodiscard]] const RealType&
   getTime() const;

   /**
    * \brief Setter of the time where the evolution computation shall by stopped.
    */
   void
   setStopTime( const RealType& stopTime );

   /**
    * \brief Getter of the time where the evolution computation shall by stopped.
    */
   [[nodiscard]] const RealType&
   getStopTime() const;

   /**
    * \brief Setter of the time step used for the computation.
    *
    * The time step can be changed by methods using adaptive choice of the time step.
    */
   void
   setTau( const RealType& tau );

   /**
    * \brief Getter of the time step used for the computation.
    */
   [[nodiscard]] const RealType&
   getTau() const;

   /**
    * \brief Setter of maximal value of the time step.
    *
    * If methods uses adaptive choice of the time step, this sets the upper limit.
    */
   void
   setMaxTau( const RealType& maxTau );

   /**
    * \brief Getter of maximal value of the time step.
    */
   [[nodiscard]] const RealType&
   getMaxTau() const;

   /**
    * \brief This method refreshes the solver monitor.
    *
    * The method propagates values of time, time step and others to the
    * solver monitor.
    */
   void
   refreshSolverMonitor( bool force = false );

   /**
    * \brief Checks if the solver is allowed to do the next iteration.
    *
    * \return true \e true if the solver is allowed to do the next iteration.
    * \return \e false if the solver is \b not allowed to do the next iteration. This may
    *    happen because the divergence occurred.
    */
   [[nodiscard]] bool
   checkNextIteration();

   void
   setTestingMode( bool testingMode );

protected:
   /****
    * Current time of the parabolic problem.
    */
   RealType time = 0.0;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau = 0.0;

   RealType maxTau = std::numeric_limits< RealType >::max();

   bool stopOnSteadyState = false;

   bool testingMode = false;
};

}  // namespace TNL::Solvers::ODE

#include <TNL/Solvers/ODE/ExplicitSolver.hpp>
