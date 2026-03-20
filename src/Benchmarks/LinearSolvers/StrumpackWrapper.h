// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_STRUMPACK
   #include <StrumpackSparseSolver.hpp>
#endif

#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>

template< typename Matrix >
class StrumpackWrapper : public TNL::Solvers::Linear::LinearSolver< Matrix >
{
   using Base = TNL::Solvers::Linear::LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using MatrixPointer = typename Base::MatrixPointer;

   StrumpackWrapper()
   {
#ifdef HAVE_STRUMPACK
      this->solver.options().set_rel_tol( 1e-10 );
      // use direct solver
      this->solver.options().set_Krylov_solver( strumpack::KrylovSolver::DIRECT );
      // enable HSS compression, see HSS Preconditioning
      this->solver.options().set_compression( strumpack::CompressionType::HSS );
      // enable GPU acceleration (oflload from host to GPU)
      //this->solver.options().enable_gpu();
#endif
   }

   void
   setMatrix( const MatrixPointer& matrix ) override
   {
#ifdef HAVE_STRUMPACK
      LinearSolver< Matrix >::setMatrix( matrix );

      this->solver.set_csr_matrix(
         matrix->getRows(),
         matrix->getSegments().getOffsets().getData(),
         matrix->getColumnIndexes().getData(),
         matrix->getValues().getData() );

      this->solver.reorder();
      this->solver.factor();
#endif
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef HAVE_STRUMPACK
      if( this->matrix->getColumns() != x.getSize() )
         throw std::invalid_argument( "StrumpackWrapper::solve: wrong size of the solution vector" );
      if( this->matrix->getColumns() != b.getSize() )
         throw std::invalid_argument( "StrumpackWrapper::solve: wrong size of the right hand side" );

      this->setResidue( NAN );
      this->solver.solve( b.getData(), x.getData() );
      this->setResidue( 0 );
      return true;
#else
      throw std::runtime_error( "StrumpackWrapper was not built with Strumpack support." );
#endif
   }

protected:
#ifdef HAVE_STRUMPACK
   strumpack::StrumpackSparseSolver< RealType, IndexType > solver;
#endif
};
