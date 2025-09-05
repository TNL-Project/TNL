// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL::Solvers::Linear {

/**
 * \brief Gaussian Elimination Method (GEM) direct solver for dense linear systems.
 *
 * This class implements the GEM algorithm for solving linear systems with dense matrices.
 * It supports optional pivoting.
 *
 * \tparam Matrix Type of the matrix representing the linear system. Must be a dense matrix.
 * \tparam Real Floating point type used for computations.
 * \tparam SolverMonitor Type of the solver monitor.
 */
template< typename Matrix, typename Real = typename Matrix::RealType, typename SolverMonitor = IterativeSolverMonitor< double > >
struct GEM : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_dense_matrix_v< Matrix >, "GEM can only be used with dense matrices." );

private:
   using Base = LinearSolver< Matrix >;
   using VectorType = typename Traits< Matrix >::VectorType;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using MatrixType = typename Base::MatrixType;
   using MatrixPointer = typename Base::MatrixPointer;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   //! \brief Default constructor.
   GEM() = default;

   /**
    * \brief Enables or disables pivoting in the GEM algorithm.
    * \param pivoting If true, pivoting is enabled; otherwise, it is disabled.
    */
   void
   setPivoting( bool pivoting );

   /**
    * \brief Checks if pivoting is enabled.
    * \return True if pivoting is enabled, false otherwise.
    */
   [[nodiscard]] bool
   getPivoting() const;

   /**
    * \brief Solves the linear system Ax = b.
    *
    * This method makes a copy of the matrix A. In case that
    * it is consuming too much memory use method solve
    * taking matrix as input parameter.
    *
    * \param b Right-hand side vector.
    * \param x Solution vector (output).
    * \return True if the solver succeeded, false otherwise.
    */
   bool
   solve( ConstVectorViewType b, VectorViewType x ) override;

   /**
    * \brief Solves the linear system Ax = b.
    *
    * \param A Matrix representing the linear system. This matrix
    *          is modified during the execution of this method.
    * \param b Right-hand side vector.
    * \param x Solution vector (output).
    * \return True if the solver succeeded, false otherwise.
    */
   bool
   solve( MatrixType& A, ConstVectorViewType b, VectorViewType x );

protected:
   /// Indicates whether pivoting is enabled.
   bool pivoting = true;
};

}  // namespace TNL::Solvers::Linear

#include "GEM.hpp"
