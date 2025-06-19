// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>

namespace TNL::Benchmarks {

template< typename Vector, typename SolverMonitor = Solvers::IterativeSolverMonitor< typename Vector::RealType > >
class MersonNonET : public Solvers::ODE::ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using DofVectorType = Vector;
   using RealType = typename Vector::RealType;
   using ValueType = RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;

   static constexpr bool
   isStatic()
   {
      return false;
   }

   MersonNonET();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setAdaptivity( const RealType& a );

   template< typename RHSFunction >
   bool
   solve( DofVectorType& u, RHSFunction&& rhsFunction );

protected:
   /****
    * The parameter u is not constant because one often
    * needs to correct u on the boundaries to be able to compute
    * the RHS.
    */
   template< typename RHSFunction >
   void
   computeKFunctions( DofVectorType& u, const RealType& time, RealType tau, RHSFunction&& rhsFunction );

   RealType
   computeError( const RealType tau );

   void
   computeNewTimeLevel( const RealType time, const RealType tau, DofVectorType& u, RealType& currentResidue );

   void
   writeGrids( const DofVectorType& u );

   DofVectorType k1, k2, k3, k4, k5, kAux;

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity;

   Containers::Vector< RealType, DeviceType, IndexType > openMPErrorEstimateBuffer;

   DofVectorType cudaBlockResidue;
};

}  // namespace TNL::Benchmarks

#include "MersonNonET.hpp"
