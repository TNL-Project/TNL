/*
 * File:   gem.h
 * Author: oberhuber
 *
 * Created on September 28, 2016, 5:30 PM
 */

#pragma once

#include <ostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Solvers/DirectSolver.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL::Solvers::Linear {

template< typename Device >
class GEMDeviceDependentCode;

template< typename Matrix, typename Real = typename Matrix::RealType, typename SolverMonitor = IterativeSolverMonitor< double > >
struct GEM : public DirectSolver< Real, typename Matrix::IndexType, SolverMonitor >
{
   static_assert( Matrices::is_dense_matrix_type_v< Matrix >, "GEM works only with dense matrices." );

   //! \brief Type for floating point numbers.
   using RealType = Real;

   //! \brief Device where the solver will run.
   using DeviceType = typename Matrix::DeviceType;

   //! \brief Indexing type.
   using IndexType = typename Matrix::IndexType;

   //! \brief Type of the matrix representing the linear system.
   using MatrixType = Matrix;

   //! \brief Type of shared pointer to the matrix.
   using MatrixPointer = std::shared_ptr< MatrixType >;

   //! \brief Type for vector representing the solution and the right-hand side.
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   GEM() = default;

   void
   setMatrix( const MatrixPointer& matrix );

   void
   setPivoting( bool pivoting );

   bool
   getPivoting() const;

   bool
   solve( const VectorType& b, VectorType& x );

#ifdef HAVE_MPI
   bool
   GEMdeviceMPI( VectorType& x, int verbose );
#endif

protected:
   void
   print( std::ostream& str = std::cout ) const;

   MatrixPointer A;

   bool pivoting = true;
};

}  // namespace TNL::Solvers::Linear
#include "GEM.hpp"
