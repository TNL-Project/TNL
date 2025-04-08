// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Optimization/LPProblem.h>

namespace TNL::Solvers::Optimization {

enum class PDLPRestarting
{
   DualityGap,
   KKTError
};

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
template< typename LPProblem_,
          typename SolverMonitor = IterativeSolverMonitor< typename LPProblem_::RealType, typename LPProblem_::IndexType > >
class PDLP : public IterativeSolver< typename LPProblem_::RealType, typename LPProblem_::IndexType, SolverMonitor >
{
public:
   using LPProblemType = LPProblem_;
   using RealType = typename LPProblemType::RealType;
   using DeviceType = typename LPProblemType::DeviceType;
   using IndexType = typename LPProblemType::IndexType;
   using MatrixType = typename LPProblemType::MatrixType;
   using VectorType = typename LPProblemType::VectorType;
   using VectorView = typename VectorType::ViewType;
   using ConstVectorView = typename VectorType::ConstViewType;

   PDLP() = default;

   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   bool
   solve( const LPProblemType& lpProblem, VectorType& x );

protected:
   bool
   adaptiveStep( const MatrixType& GA,
                 const MatrixType& GAT,
                 const VectorType& q,
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

   RealType
   KKTError( const MatrixType& GA,
             const MatrixType& GAT,
             const IndexType m1,
             const VectorType& q,
             const VectorView& x,
             const VectorView& y,
             const VectorType& u,
             const VectorType& l,
             const VectorType& c,
             const RealType& omega ) const;

   RealType
   primalDualGap( const MatrixType& GA,
                  const MatrixType& GAT,
                  const IndexType m1,
                  const VectorType& c,
                  const VectorType& q,
                  const VectorType& u,
                  const VectorType& l,
                  const VectorView& z,
                  const VectorView& z_ref ) const;

   VectorType primal_gradient;

   PDLPRestarting restarting = PDLPRestarting::DualityGap;  //KKTError;
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/PDLP.hpp>
