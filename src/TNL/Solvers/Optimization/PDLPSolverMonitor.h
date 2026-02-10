// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/SolverMonitor.h>

#include "KKTData.h"

namespace TNL::Solvers::Optimization {

enum class ConvergenceGraphType
{
   NoConvergenceGraphs,
   WriteConvergenceGraphs
};

enum class RestartingType
{
   Artificial,
   Sufficient,
   Necessary
};

enum class RestartingTo
{
   Average,
   Current
};

template< typename Real = double >
struct PDLPSolverMonitor : public SolverMonitor
{
   using RealType = Real;
   using KKTDataType = KKTData< RealType >;

   PDLPSolverMonitor() = default;

   void
   setConvergenceGraphs( ConvergenceGraphType type )
   {
      this->convergenceGraphType = type;
      if( type == ConvergenceGraphType::WriteConvergenceGraphs ) {
         // Open files for writing convergence graphs
         kkt_current_primal_objective_file.open( "kkt-current-primal-objective.txt", std::ios::out );
         kkt_current_dual_objective_file.open( "kkt-current-dual-objective.txt", std::ios::out );
         kkt_averaged_primal_objective_file.open( "kkt-averaged-primal-objective.txt", std::ios::out );
         kkt_averaged_dual_objective_file.open( "kkt-averaged-dual-objective.txt", std::ios::out );
         kkt_current_duality_gap_file.open( "kkt-current-duality-gap.txt", std::ios::out );
         kkt_averaged_duality_gap_file.open( "kkt-averaged-duality-gap.txt", std::ios::out );
         kkt_current_primal_feasibility_file.open( "kkt-current-primal-feasibility.txt", std::ios::out );
         kkt_current_dual_feasibility_file.open( "kkt-current-dual-feasibility.txt", std::ios::out );
         kkt_averaged_primal_feasibility_file.open( "kkt-averaged-primal-feasibility.txt", std::ios::out );
         kkt_averaged_dual_feasibility_file.open( "kkt-averaged-dual-feasibility.txt", std::ios::out );
         kkt_current_mu_file.open( "kkt-current-mu.txt", std::ios::out );
         kkt_averaged_mu_file.open( "kkt-averaged-mu.txt", std::ios::out );
         restarts_file.open( "restarts.txt", std::ios::out );
      }
   }

   ConvergenceGraphType
   getConvergenceGraphs() const
   {
      return this->convergenceGraphType;
   }

   void
   setCurrentKKT( const KKTDataType& kktData )
   {
      current_kkt_data = kktData;
   }

   void
   setAveragedKKT( const KKTDataType& kktData )
   {
      averaged_kkt_data = kktData;
   }

   const KKTDataType&
   getCurrentKKT() const
   {
      return current_kkt_data;
   }

   const KKTDataType&
   getAveragedKKT() const
   {
      return averaged_kkt_data;
   }

   void
   setCurrentPrimalDualGap( const RealType& gap )
   {
      current_primal_dual_gap = gap;
   }

   void
   setAveragedPrimalDualGap( const RealType& gap )
   {
      averaged_primal_dual_gap = gap;
   }

   const RealType&
   getCurrentPrimalDualGap() const
   {
      return current_primal_dual_gap;
   }

   const RealType&
   getAveragedPrimalDualGap() const
   {
      return averaged_primal_dual_gap;
   }

   void
   setRestarting( RestartingType restarting, RestartingTo candidate )
   {
      this->restarting = restarting;
      this->restartingTo = candidate;
   }

   void
   refresh() override
   {}

protected:
   KKTDataType current_kkt_data;
   KKTDataType averaged_kkt_data;
   RealType current_primal_dual_gap;
   RealType averaged_primal_dual_gap;

   ConvergenceGraphType convergenceGraphType = ConvergenceGraphType::NoConvergenceGraphs;
   RestartingType restarting = RestartingType::Artificial;
   RestartingTo restartingTo = RestartingTo::Current;

   // Files for writing convergence graphs
   std::fstream kkt_current_primal_objective_file;
   std::fstream kkt_current_dual_objective_file;
   std::fstream kkt_averaged_primal_objective_file;
   std::fstream kkt_averaged_dual_objective_file;
   std::fstream kkt_current_duality_gap_file;
   std::fstream kkt_averaged_duality_gap_file;
   std::fstream kkt_current_primal_feasibility_file;
   std::fstream kkt_current_dual_feasibility_file;
   std::fstream kkt_averaged_primal_feasibility_file;
   std::fstream kkt_averaged_dual_feasibility_file;
   std::fstream kkt_current_mu_file;
   std::fstream kkt_averaged_mu_file;
   std::fstream fast_current_primal_objective_file;
   std::fstream fast_current_dual_objective_file;
   std::fstream fast_averaged_primal_objective_file;
   std::fstream fast_averaged_dual_objective_file;
   std::fstream fast_current_duality_gap_file;
   std::fstream fast_averaged_duality_gap_file;
   std::fstream fast_current_primal_feasibility_file;
   std::fstream fast_current_dual_feasibility_file;
   std::fstream fast_current_mu_file;
   std::fstream fast_averaged_mu_file;
   std::fstream current_gradient_file;
   std::fstream averaged_gradient_file;
   std::fstream restarts_file;
};

}  // namespace TNL::Solvers::Optimization
