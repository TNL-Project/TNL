// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( HAVE_GINKGO )

   #include <memory>

   #include <ginkgo/ginkgo.hpp>

   #include "ReferenceKernels.h"

namespace TNL::Benchmarks::SpMM {

/**
 * \brief Wrapper of the product of a `gko::matrix::Csr` and a `gko::matrix::Dense`
 * with the OpenMP executor of Ginkgo.
 *
 * The Ginkgo matrices are views of the arrays of the benchmark, so that
 * \ref multiply measures only the multiplication itself. The dense matrices
 * of Ginkgo are row-major, so only the row-major layout is supported.
 */
template< typename Real, typename Index >
class GinkgoSpMM
{
public:
   GinkgoSpMM(
      const CSRArrays< Real, Index >& A,
      const DenseArray< Real, Index >& B,
      DenseArray< Real, Index >& C,
      gko::matrix::csr::spmv_strategy strategy )
   {
      const auto exec = gko::OmpExecutor::create();
      matA = gko::matrix::Csr< Real, Index >::create_const(
         exec,
         gko::dim< 2 >( A.rows, A.columns ),
         gko::make_const_array_view( exec, A.nonzeros, A.values ),
         gko::make_const_array_view( exec, A.nonzeros, A.columnIndexes ),
         gko::make_const_array_view( exec, A.rows + 1, A.rowPointers ),
         strategy );
      matB = gko::matrix::Dense< Real >::create_const(
         exec,
         gko::dim< 2 >( B.rows, B.columns ),
         gko::make_const_array_view( exec, ( B.rows - 1 ) * B.leadingDimension + B.columns, B.data ),
         B.leadingDimension );
      matC = gko::matrix::Dense< Real >::create(
         exec,
         gko::dim< 2 >( C.rows, C.columns ),
         gko::make_array_view( exec, ( C.rows - 1 ) * C.leadingDimension + C.columns, C.data ),
         C.leadingDimension );
   }

   void
   multiply()
   {
      matA->apply( matB, matC );
   }

private:
   std::unique_ptr< const gko::matrix::Csr< Real, Index > > matA;
   std::unique_ptr< const gko::matrix::Dense< Real > > matB;
   std::unique_ptr< gko::matrix::Dense< Real > > matC;
};

}  // namespace TNL::Benchmarks::SpMM

#endif
