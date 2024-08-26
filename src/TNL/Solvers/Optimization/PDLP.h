// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL::Solvers::Optimization {

/**
 * \brief Implementation of Primal-Dual Hybrid Gradient Method for Linear Programming (PDLP).
 *
 * See the following paper for more details:
 *
 * Applegate, David, et al. "Practical large-scale linear programming using primal-dual hybrid gradient." Advances in Neural
 * Information Processing Systems 34 (2021): 20243-20257.
 *
 * https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf
 *
 */
template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class PDLP : public IterativeSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorType = Vector;
   using VectorView = typename Vector::ViewType;
   using ConstVectorView = typename Vector::ConstViewType;

   PDLP() = default;

   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   template< typename MatrixType >
   bool
   solve( const Vector& c,
          const MatrixType& GA,
          const Vector& hb,
          const IndexType m1,
          const Vector& l,
          const Vector& u,
          Vector& x );

protected:
   template< typename MatrixType >
   bool
   adaptiveStep( const MatrixType& GA,
                 const MatrixType& GAT,
                 const VectorType& hb,
                 const IndexType m1,
                 const VectorType& u,
                 const VectorType& l,
                 const VectorType& c,
                 const VectorView& in_x,
                 const VectorView& in_y,
                 VectorView& out_x,
                 VectorView& out_y,
                 IndexType k,
                 RealType& current_omega,
                 RealType& current_eta );
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/PDLP.hpp>
