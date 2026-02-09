// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Solvers::Optimization {

template< typename Real >
struct KKTData
{
   Real primal_feasibility;
   Real dual_feasibility;
   Real primal_objective;
   Real dual_objective;

   Real
   getPrimalFeasibility() const
   {
      return primal_feasibility;
   }

   Real
   getDualFeasibility() const
   {
      return dual_feasibility;
   }

   Real
   getPrimalObjective() const
   {
      return primal_objective;
   }

   Real
   getDualObjective() const
   {
      return dual_objective;
   }

   Real
   getDualityGap() const
   {
      return std::abs( primal_objective - dual_objective );
   }

   Real
   getKKTError( const Real& omega ) const;

   Real
   getRelativeDualityGap() const;
};

}  // namespace TNL::Solvers::Optimization
