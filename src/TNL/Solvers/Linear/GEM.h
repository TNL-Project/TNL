/*
 * File:   gem.h
 * Author: oberhuber
 *
 * Created on September 28, 2016, 5:30 PM
 */

#pragma once

#include <ostream>
#include <TNL/Matrices/DenseMatrix.h>

namespace TNL::Solvers::Linear {

template< typename Device >
class GEMDeviceDependentCode;

template< typename Matrix >
struct GEM
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using MatrixGEM = Matrix;                                                       // TODO: Rename to MatrixType
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;  // TODO: Rename to VectorType

   GEM( MatrixGEM& A, VectorType& b );

   bool
   solve( VectorType& x, int verbose = 0 );

   bool
   solveWithoutPivoting( VectorType& x, int verbose = 0 );

   bool
   solveWithPivoting( VectorType& x, int verbose = 0 );

   bool
   computeLUDecomposition( int verbose = 0 );

#ifdef HAVE_MPI
   bool
   GEMdeviceMPI( VectorType& x, int verbose );
#endif

   bool
   GEMdevice( VectorType& x, int verbose );

   bool
   setMatrixVector( MatrixGEM& A, VectorType& b )
   {
      this->A = A;
      this->b = b;
      return true;
   }

   void
   setPivoting( bool pivoting )
   {
      this->pivoting = pivoting;
   }

   bool
   getPivoting() const
   {
      return this->pivoting;
   }

protected:
   void
   print( std::ostream& str = std::cout ) const;

   MatrixGEM A;

   VectorType b;

   bool pivoting = true;
   //typedef GEMDeviceDependentCode< DeviceType > DeviceDependentCode;
   //friend class GEMDeviceDependentCode< DeviceType >;
};

}  // namespace TNL::Solvers::Linear
#include "GEM.hpp"
