// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Optimization/LPProblem.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRVectorKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRLightKernel.h>
#include <TNL/Matrices/CusparseCSRMatrix.h>

#include "KKTData.h"
#include "PDLPSolverMonitor.h"

namespace TNL::Solvers::Optimization {

enum class PDLPRestarting
{
   None,
   Constant,
   DualityGap,
   KKT,
   Fast
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
template< typename LPProblem_, typename SolverMonitor = IterativeSolverMonitor< typename LPProblem_::RealType > >
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
   KKT( const VectorType& z, const VectorType& Kz );

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
   computeLambda( const VectorType& c, const ConstVectorView& KTy, const VectorType& l, const VectorType& u, VectorType& lambda );

   RealType
   computePrimalFeasibility( const VectorType& q, const ConstVectorView& Kx );

protected:
   RealType
   primalDualGap( const VectorView& z, const VectorView& z_ref );

   void
   printReport( const KKTDataType& kkt, IndexType iterations, bool converged = false );

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
   VectorType Kz, Kz_current, Kz_averaged, Kz_candidate;
   VectorType Kx, KTy;
   VectorType lambda;

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

   SegmentsReductionKernel segmentsReductionKernel;

   bool useCusparse = false;  // Use cusparse for matrix-vector product
   Matrices::CusparseCSRMatrix< MatrixType > cusparseK;
   Matrices::CusparseCSRMatrix< MatrixType > cusparseKT;

   PDLPSolverMonitor< RealType > monitor;
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/PDLP.hpp>
