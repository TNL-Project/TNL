// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/SolverMonitor.h>

namespace TNL::Solvers {

/**
 * \brief Object for monitoring direct solvers.
 *
 * \tparam Real is a type of the floating-point arithmetics.
 * \tparam Index is an indexing type.
 *
 * The following example shows how to use the direct solver monitor for monitoring
 * convergence of linear direct solver:
 *
 * \includelineno Solvers/Linear/DirectLinearSolverWithMonitorExample.cpp
 *
 * The result looks as follows:
 *
 * \include DirectLinearSolverWithMonitorExample.out
 *
 * The following example shows how to employ timer (\ref TNL::Timer) to the monitor
 * of direct solvers:
 *
 * \includelineno Solvers/Linear/DirectLinearSolverWithTimerExample.cpp
 *
 * The result looks as follows:
 *
 * \include DirectLinearSolverWithTimerExample.out
 */
template< typename Real = double, typename Index = int >
class DirectSolverMonitor : public SolverMonitor
{
public:
   /**
    * \brief A type of the floating-point arithmetics.
    */
   using RealType = Real;

   /**
    * \brief A type for indexing.
    */
   using IndexType = Index;

   /**
    * \brief Construct with no parameters.
    */
   DirectSolverMonitor() = default;

   /**
    * \brief This method can be used for naming a stage of the monitored solver.
    *
    * The stage name can be used to differ between various stages of direct solvers.
    *
    * \param stage is name of the solver stage.
    */
   void
   setStage( const std::string& stage );

   /**
    * \brief Set up the verbosity of the monitor.
    *
    * \param verbose is the new value of the verbosity of the monitor.
    */
   void
   setVerbose( const IndexType& verbose );

   /**
    * \brief Causes that the monitor prints out the status of the solver.
    */
   void
   refresh() override;

protected:
   int
   getLineWidth();

   std::string stage, saved_stage;

   std::atomic_bool saved{ false };
   std::atomic_bool attributes_changed{ false };

   RealType elapsed_time_before_refresh = 0;

   IndexType verbose = 2;
};

}  // namespace TNL::Solvers

#include <TNL/Solvers/DirectSolverMonitor.hpp>
