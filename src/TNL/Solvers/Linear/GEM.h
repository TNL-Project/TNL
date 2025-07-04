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
   using RealType = Real;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using MatrixType = Matrix;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   GEM( MatrixType& A );

   bool
   solve( const VectorType& b, VectorType& x );

#ifdef HAVE_MPI
   bool
   GEMdeviceMPI( VectorType& x, int verbose );
#endif

   bool
   setMatrixVector( MatrixType& A, VectorType& b )
   {
      this->A = A;
      this->b = b;
      return true;
   }

   void
   setPivoting( bool pivoting );

   bool
   getPivoting() const;

protected:
   void
   print( std::ostream& str = std::cout ) const;

   MatrixType A;

   VectorType b;

   bool pivoting = true;
};

}  // namespace TNL::Solvers::Linear
#include "GEM.hpp"
