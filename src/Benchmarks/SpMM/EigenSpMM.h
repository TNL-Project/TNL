// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( HAVE_EIGEN )

   #include <type_traits>

   #include <Eigen/Core>
   #include <Eigen/SparseCore>

   #include "ReferenceKernels.h"

namespace TNL::Benchmarks::SpMM {

/**
 * \brief Product of a CSR matrix and a dense matrix computed by Eigen.
 *
 * The arrays are only mapped by `Eigen::Map`, nothing is copied. Eigen
 * parallelizes the product of a row-major sparse matrix and a dense matrix
 * over the rows with OpenMP.
 */
template< typename Real, typename Index >
void
eigenSpMM( const CSRArrays< Real, Index >& A, const DenseArray< Real, Index >& B, DenseArray< Real, Index >& C, bool rowMajor )
{
   using SparseMatrix = Eigen::SparseMatrix< Real, Eigen::RowMajor, Index >;
   const Eigen::Map< const SparseMatrix > matA(
      A.rows, A.columns, A.nonzeros, A.rowPointers, A.columnIndexes, A.values );
   auto multiply = [ & ]( auto storageOrder )
   {
      using DenseMatrix = Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic, decltype( storageOrder )::value >;
      using Stride = Eigen::OuterStride<>;
      const Eigen::Map< const DenseMatrix, 0, Stride > matB( B.data, B.rows, B.columns, Stride( B.leadingDimension ) );
      Eigen::Map< DenseMatrix, 0, Stride > matC( C.data, C.rows, C.columns, Stride( C.leadingDimension ) );
      matC.noalias() = matA * matB;
   };
   if( rowMajor )
      multiply( std::integral_constant< int, Eigen::RowMajor >{} );
   else
      multiply( std::integral_constant< int, Eigen::ColMajor >{} );
}

}  // namespace TNL::Benchmarks::SpMM

#endif
