// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL::Solvers::Optimization {

template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class NesterovMomentum : public IterativeSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorType = Vector;
   using VectorView = typename Vector::ViewType;

   NesterovMomentum() = default;

   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   void
   setRelaxation( const RealType& lambda );

   const RealType&
   getRelaxation() const;

   void
   setMomentum( const RealType& beta );

   const RealType&
   getMomentum() const;

   template< typename GradientGetter >
   bool
   solve( VectorView& w, GradientGetter&& getGradient );

protected:
   RealType relaxation = 1.0, momentum = 0.9;

   VectorType gradient, v, aux;
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/NesterovMomentum.hpp>
