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
   static_assert( Matrices::is_dense_matrix_v< Matrix >, "GEM works only with dense matrices." );

   using Base = LinearSolver< Matrix >;

   //! \brief Type for floating point numbers.
   using RealType = typename Base::RealType;

   //! \brief Device where the solver will run.
   using DeviceType = typename Base::DeviceType;

   //! \brief Indexing type.
   using IndexType = typename Base::IndexType;

   //! \brief Type of the matrix representing the linear system.
   using MatrixType = typename Base::MatrixType;

   //! \brief Type of shared pointer to the matrix.
   using MatrixPointer = std::shared_ptr< MatrixType >;

   //! \brief Type for vector representing the solution and the right-hand side.
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   //! \brief Type for vector view.
   using VectorViewType = typename Base::VectorViewType;

   //! \brief Type for constant vector view.
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   //! \brief Default constructor.
   GEM() = default;

   /**
    * \brief Sets the matrix representing the linear system.
    * \param matrix Shared pointer to the matrix.
    */
   void
   setMatrix( const MatrixPointer&
                 matrix );  // This does not override LinearSolver::setMatrix since we need to pass non-constant matrix here

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
    * \param b Right-hand side vector.
    * \param x Solution vector (output).
    * \return True if the solver succeeded, false otherwise.
    */
   bool
   solve( ConstVectorViewType b, VectorViewType x ) override;

   /**
    * \brief Checks if the last solve operation was successful.
    * \return True if the solver succeeded, false otherwise.
    */
   [[nodiscard]] bool
   succeeded() const;

protected:
   void
   print( std::ostream& str = std::cout ) const;

   /// Shared pointer to the matrix representing the linear system.
   MatrixPointer A;

   /// Indicates whether the last solve operation was successful.
   bool success = false;

   /// Indicates whether pivoting is enabled.
   bool pivoting = true;
};

}  // namespace TNL::Solvers::Linear

#include "GEM.hpp"
