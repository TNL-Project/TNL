// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>

#include "../../BLAS/CommonVectorOperations.h"

namespace TNL {
namespace Benchmarks {

template< typename Vector,
          typename SolverMonitor = Solvers::IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class EulerNonET : public Solvers::ODE::ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
   public:
   using DofVectorType = Vector;
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorOperations = CommonVectorOperations< DeviceType >;

   EulerNonET();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setCFLCondition( const RealType& cfl );

   const RealType&
   getCFLCondition() const;

   template< typename RHSFunction >
   bool
   solve( DofVectorType& u, RHSFunction&& rhsFunction );


   protected:
   void computeNewTimeLevel( DofVectorType& u,
                             RealType tau,
                             RealType& currentResidue );

   DofVectorType k1;

   RealType cflCondition;

   DofVectorType cudaBlockResidue;
};

}  // namespace Benchmarks
}  // namespace TNL

#include "EulerNonET.hpp"
