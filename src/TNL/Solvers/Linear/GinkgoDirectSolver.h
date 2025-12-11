// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_GINKGO
   #include <ginkgo/ginkgo.hpp>
   #include <TNL/Matrices/GinkgoOperator.h>
   #include <TNL/Containers/GinkgoVector.h>
#endif

#include "LinearSolver.h"
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Algorithms/Segments/CSR.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
class GinkgoDirectSolver : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_sparse_csr_matrix_v< Matrix >, "Umfpack works only with CSR format." );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using MatrixPointer = typename Base::MatrixPointer;

   GinkgoDirectSolver()
   {
#ifdef HAVE_GINKGO
      if( std::is_same_v< DeviceType, TNL::Devices::Host > )
         gk_exec = gko::OmpExecutor::create();
      if( std::is_same_v< DeviceType, TNL::Devices::Cuda > )
         gk_exec = gko::CudaExecutor::create( 0, gko::OmpExecutor::create() );
      //if( std::is_same_v< DeviceType, TNL::Devices::Hip > ) // This is true even for CUDA!!!!!
      //   gk_exec = gko::HipExecutor::create( 0, gko::OmpExecutor::create() );
#else
      throw std::runtime_error( "GinkgoDirectSolver was not built with Ginkgo support." );
#endif
   }

   void
   setMatrix( const MatrixPointer& matrix ) override
   {
#ifdef HAVE_GINKGO
      this->matrix = matrix;
      auto gko_A = gko::share( gko::matrix::Csr< RealType, IndexType >::create(
         gk_exec,
         gko::dim< 2 >{ static_cast< std::size_t >( matrix->getRows() ), static_cast< std::size_t >( matrix->getColumns() ) },
         gko::make_array_view(
            gk_exec, matrix->getNonzeroElementsCount(), const_cast< RealType* >( matrix->getValues().getData() ) ),
         gko::make_array_view(
            gk_exec, matrix->getNonzeroElementsCount(), const_cast< IndexType* >( matrix->getColumnIndexes().getData() ) ),
         gko::make_array_view(
            gk_exec, matrix->getRows() + 1, const_cast< IndexType* >( matrix->getSegments().getOffsets().getData() ) ) ) );

      gk_solver = gko::experimental::solver::Direct< RealType, IndexType >::build()
                     .with_factorization( gko::experimental::factorization::Lu< RealType, IndexType >::build() )
                     .on( gk_exec )
                     ->generate( gko_A );
      // See https://github.com/ginkgo-project/ginkgo/discussions/1637 for details how to perform symbolic factorization first
      // followed by the numerical one.
#else
      throw std::runtime_error( "GinkgoDirectSolver was not built with Ginkgo support." );
#endif
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef HAVE_GINKGO
      this->resetIterations();
      this->setResidue( NAN );
      auto gko_b = gko::matrix::Dense< RealType >::create(
         gk_exec,
         gko::dim< 2 >{ static_cast< std::size_t >( b.getSize() ), 1 },
         gko::make_array_view( gk_exec, b.getSize(), const_cast< RealType* >( b.getData() ) ),
         1 );
      auto gko_x = gko::matrix::Dense< RealType >::create(
         gk_exec,
         gko::dim< 2 >{ static_cast< std::size_t >( x.getSize() ), 1 },
         gko::make_array_view( gk_exec, x.getSize(), const_cast< RealType* >( x.getData() ) ),
         1 );

      gk_solver->apply( gko_b, gko_x );
      this->setResidue( 0 );
      return true;
#else
      throw std::runtime_error( "GinkgoDirectSolver was not built with Ginkgo support." );
#endif
   }

#ifdef HAVE_GINKGO
   std::shared_ptr< gko::Executor > gk_exec;
   std::shared_ptr< gko::experimental::solver::Direct< RealType, IndexType > > gk_solver;
#endif
};

}  // namespace TNL::Solvers::Linear
