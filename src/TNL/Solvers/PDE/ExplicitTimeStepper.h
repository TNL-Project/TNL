// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename DofVector, template< typename DofVector_, typename SolverMonitor > class OdeSolver >
class ExplicitTimeStepper
{
public:
   using RealType = typename DofVector::RealType;
   using DeviceType = typename DofVector::DeviceType;
   using IndexType = typename DofVector::IndexType;
   using DofVectorType = DofVector;
   // using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using SolverMonitorType = IterativeSolverMonitor< RealType, IndexType >;
   using OdeSolverType = OdeSolver< DofVectorType, SolverMonitorType >;
   // using OdeSolverPointer = Pointers::SharedPointer< OdeSolverType, DeviceType >;

   ExplicitTimeStepper() = default;

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   bool
   init( const MeshType& mesh );

   bool
   init();

   void
   setSolverMonitor( SolverMonitorType& solverMonitor );

   void
   setProblem( ProblemType& problem );

   // void setProblem( ProblemType& problem );

   // ProblemType* getProblem() const;

   bool
   solve( const RealType& time, const RealType& stopTime, DofVectorPointer& dofVector );

   bool
   solve( const RealType& time, const RealType& stopTime, DofVectorType& dofVector );

   void
   getExplicitUpdate( const RealType& time, const RealType& tau, DofVectorPointer& _u, DofVectorPointer& _fu );

   void
   getExplicitUpdate( const RealType& time, const RealType& tau, DofVectorType& _u, DofVectorType& _fu );

   void
   applyBoundaryConditions( const RealType& time, DofVectorType& _u );

protected:
   OdeSolverPointer odeSolver;

   SolverMonitorType* solverMonitor;

   OdeSolverType odeSolver;

   RealType timeStep;

   // Problem* problem;

   RealType timeStep = 0.0;

   Timer preIterateTimer, explicitUpdaterTimer, mainTimer, postIterateTimer;

   long long int allIterations = 0;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/PDE/ExplicitTimeStepper.hpp>
