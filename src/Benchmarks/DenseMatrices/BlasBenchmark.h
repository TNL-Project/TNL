// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixBase.h>
#include <type_traits>

#ifdef HAVE_BLAS

   #include <cblas.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationBLAS( const DenseMatrix& matrix1, const DenseMatrix& matrix2, DenseMatrix& resultMatrix )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Host >, "This function is specialized for Host device only." );

   IndexType n = matrix2.getColumns();
   IndexType k = matrix1.getColumns();
   IndexType m = matrix1.getRows();

   auto organization = matrix1.getOrganization();

   if constexpr( std::is_same_v< RealType, float > ) {
      cblas_sgemm( organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
                   CblasNoTrans,
                   CblasNoTrans,
                   m,
                   n,
                   k,
                   1.0f,
                   matrix1.getValues().getData(),
                   k,
                   matrix2.getValues().getData(),
                   n,
                   0.0f,
                   resultMatrix.getValues().getData(),
                   organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      cblas_dgemm( organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
                   CblasNoTrans,
                   CblasNoTrans,
                   m,
                   n,
                   k,
                   1.0,
                   matrix1.getValues().getData(),
                   k,
                   matrix2.getValues().getData(),
                   n,
                   0.0,
                   resultMatrix.getValues().getData(),
                   organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m );
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices

#endif  //HAVE_BLAS
