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

   std::tuple< bool, RealType, RealType >
   solve( const LPProblemType& lpProblem, VectorType& x );

protected:
   void
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

   /**
    * \brief Computes the KKT error.
    *
    * \param GA The constraint matrix.
    * \param GAT The transposed constraint matrix.
    * \param m1 The number of inequality constraints.
    * \param c The objective function coefficients.
    * \param q The right-hand side vector.
    * \param z The primal-dual variable vector.
    * \param u The upper bounds vector.
    * \param l The lower bounds vector.
    *
    * \return A tuple containing the primal feasibility, dual feasibility, primal objective and dual objective.
    */
   std::tuple< RealType, RealType, RealType, RealType >
   KKTError( const MatrixType& GA,
             const MatrixType& GAT,
             const IndexType m1,
             const VectorType& c,
             const VectorType& q,
             const VectorView& z,
             const VectorType& u,
             const VectorType& l ) const;

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
