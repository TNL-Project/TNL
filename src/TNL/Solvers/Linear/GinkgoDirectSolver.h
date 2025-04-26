// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_GINKGO

   #include <ginkgo/ginkgo.hpp>

   #include "LinearSolver.h"
   #include <TNL/Matrices/SparseMatrix.h>
   #include <TNL/Algorithms/Segments/CSR.h>
   #include <TNL/Matrices/GinkgoOperator.h>
   #include <TNL/Containers/GinkgoVector.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
class GinkgoDirectSolver : public LinearSolver< Matrix >
{
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
      if( std::is_same_v< DeviceType, TNL::Devices::Host > )
         gk_exec = gko::OmpExecutor::create();
      if( std::is_same_v< DeviceType, TNL::Devices::Cuda > )
         gk_exec = gko::CudaExecutor::create( 0, gko::OmpExecutor::create() );
      //if( std::is_same_v< DeviceType, TNL::Devices::Hip > ) // This is true even for CUDA!!!!!
      //   gk_exec = gko::HipExecutor::create( 0, gko::OmpExecutor::create() );
   }

   void
   setMatrix( const MatrixPointer& matrix ) override
   {
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
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
      auto gko_b = gko::matrix::Dense< RealType >::create(
         gk_exec,
         gko::dim< 2 >{ static_cast< std::size_t >( b.getSize() ), 1 },
         gko::make_array_view( gk_exec, b.getSize(), const_cast< RealType* >( b.getData() ) ),
         1 );
      auto gko_x = TNL::Containers::GinkgoVector< RealType, DeviceType >::create( gk_exec, x.getView() );

      gk_solver->apply( gko_b, gko_x );
      return true;
   }

   std::shared_ptr< gko::Executor > gk_exec;
   std::shared_ptr< gko::experimental::solver::Direct< RealType, IndexType > > gk_solver;
};

}  // namespace TNL::Solvers::Linear

//#include "GinkgoDirectSolver.hpp"

#endif
