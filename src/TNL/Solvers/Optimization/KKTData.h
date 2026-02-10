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

   KKTData() = default;

   KKTData( Real primal_feasibility, Real dual_feasibility, Real primal_objective, Real dual_objective )
   : primal_feasibility( primal_feasibility ),
     dual_feasibility( dual_feasibility ),
     primal_objective( primal_objective ),
     dual_objective( dual_objective )
   {}

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
   getKKTError( const Real& omega ) const
   {
      const Real omega_sqr = omega * omega;
      const Real error =
         sqrt( omega_sqr * primal_feasibility * primal_feasibility + 1.0 / omega_sqr * ( dual_feasibility * dual_feasibility )
               + pow( primal_objective - dual_objective, 2 ) );

#ifdef PRINTING
      std::cout << " omega sqr. = " << omega_sqr << " primal feas. = " << primal_feasibility
                << " dual feas. = " << dual_feasibility << " primal obj. = " << primal_objective
                << " dual obj. = " << dual_objective << " error = " << error << std::endl;
#endif
      return error;
   }

   Real
   getRelativeDualityGap() const
   {
      return std::abs( ( primal_objective - dual_objective )
                       / ( (Real) 1.0 + std::abs( primal_objective ) + std::abs( dual_objective ) ) );
   }
};

}  // namespace TNL::Solvers::Optimization
