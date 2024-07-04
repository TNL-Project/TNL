// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_STRUMPACK
   #include <StrumpackSparseSolver.hpp>
#endif

#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/GinkgoOperator.h>
#include <TNL/Containers/GinkgoVector.h>

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
      this->solver.options().set_Krylov_solver( strumpack::KrylovSolver::DIRECT );  // use direct solver
      this->solver.options().set_compression(
         strumpack::CompressionType::HSS );  // enable HSS compression, see HSS Preconditioning
                                             //this->solver.options().enable_gpu();  // this will oflload to GPU
#endif
   }

   void
   setMatrix( const MatrixPointer& matrix )
   {
#ifdef HAVE_STRUMPACK
      this->solver.set_csr_matrix( matrix->getRows(),
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
      this->solver.solve( b.getData(), x.getData() );
      return true;
#else
      std::cerr << "Strumpack is not available." << std::endl;
      return false;
#endif
   }

protected:
   strumpack::StrumpackSparseSolver< RealType, IndexType > solver;
};
