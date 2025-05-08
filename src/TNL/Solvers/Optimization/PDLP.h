// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Optimization/LPProblem.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRVectorKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRLightKernel.h>

namespace TNL::Solvers::Optimization {

enum class PDLPRestarting
{
   None,
   Constant,
   DualityGap,
   KKT,
   Fast
};

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
   using KKTDataType = KKTData< RealType >;
   using SegmentsReductionKernel = typename MatrixType::DefaultSegmentsReductionKernel;
   //using SegmentsReductionKernel = typename Algorithms::SegmentsReductionKernels::CSRLightKernel< IndexType, DeviceType >;

   PDLP() = default;

   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   std::tuple< bool, RealType, RealType >
   solve( const LPProblemType& lpProblem, VectorType& x );

   void
   setInequalitiesFirst( bool inequalitiesFirst )
   {
      this->inequalitiesFirst = inequalitiesFirst;
   }

   void
   setRestarting( PDLPRestarting restarting )
   {
      this->restarting = restarting;
   }

   PDLPRestarting
   getRestarting() const
   {
      return this->restarting;
   }

   void
   setMaximalRestartingInterval( IndexType maxRestartingInterval )
   {
      this->maxRestartingInterval = maxRestartingInterval;
   }

protected:
   std::tuple< bool, RealType, RealType >
   PDHG( VectorType& x, VectorType& y );

   void
   adaptiveStep( const VectorType& in_z, VectorType& out_z, IndexType k, RealType& current_omega, RealType& current_eta );

   /**
    * \brief Computes the KKT error.
    *
    * \param z The primal-dual variable vector.
    *
    * \return A tuple containing the primal feasibility, dual feasibility, primal objective and dual objective.
    */
   KKTDataType
   KKT( const VectorView& z, const VectorType& Kx, const VectorType& KTy );

   void
   computeKx( const ConstVectorView& x, VectorView& Kx );

   void
   computeKTy( const ConstVectorView& y, VectorView& KTy );

   void
   computePrimalStep( const ConstVectorView& x, const VectorView& KTy, const RealType& tau, VectorView& x_new );

   void
   computeDualStep( const ConstVectorView& y,
                    const VectorView& Kx,
                    const VectorView& Kx_new,
                    const RealType& sigma,
                    VectorView& y_new );

public:  // TODO: Just because of nvcc
   void
   computeLambda( const VectorType& c, const VectorType& KTy, const VectorType& l, const VectorType& u, VectorType& lambda );

   RealType
   computePrimalFeasibility( const VectorType& q, const VectorType& Kx );

protected:
   RealType
   primalDualGap( const VectorView& z, const VectorView& z_ref );

   MatrixType K;           // The constraints matrix.   TODO: Make this matrix view
   MatrixType KT;          // The transposed constraints matrix.
   VectorType c;           // The objective function coefficients.
   VectorType q;           // The right-hand side vector.
   VectorType l;           // The lower bounds vector.
   VectorType u;           // The upper bounds vector.
   VectorType filtered_l;  // The filtered lower bounds vector.
   VectorType filtered_u;  // The filtered upper bounds vector.

   IndexType n;   // Number of variables
   IndexType m1;  // Number of equality/inequality constraints depending on the order of the matrix
   IndexType m2;  // Number of equality/inequality constraints depending on the order of the matrix
   IndexType m;   // Number of constraints
   IndexType N;   // Number of variables + constraints

   PDLPRestarting restarting = PDLPRestarting::KKT;
   IndexType maxRestartingInterval = 0;  // Maximal interval without restarting

   // Restart criteria
   RealType beta_sufficient = 0.2;   // This is used in cuPDLP-C
   RealType beta_necessary = 0.8;    // This is used in cuPDLP-C
   RealType beta_artificial = 0.36;  // This is used in cuPDLP-C
   //RealType beta_sufficient = 0.9;  // This is used in the original paper
   //RealType beta_necessary = 0.1;  // This is used in the original paper
   //RealType beta_artificial = 0.5;  // This is used in the original paper

   // Preconditioning
   VectorType D1, D2;

   // Supporting vectors
   VectorType Kx, Kx_new, Kx_averaged, KTy, KTy_averaged, lambda;

   RealType K_norm;

   IndexType adaptive_k = 1;
   bool inequalitiesFirst = true;

   bool averaging = true;
   bool adaptivePrimalWeight = true;

   // Performance measuring
   Timer solverTimer, spmvTimer;

   IndexType KxComputations = 0;
   IndexType KTyComputations = 0;

   // Convergence logging
   bool writeConvergenceGraphs = false;
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

   SegmentsReductionKernel segmentsReductionKernel;
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/PDLP.hpp>
