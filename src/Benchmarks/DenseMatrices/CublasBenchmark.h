// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__

   #include <cublas_v2.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationCuBLAS( const DenseMatrix& matrix1,
                            const DenseMatrix& matrix2,
                            DenseMatrix& resultMatrix,
                            bool transposeA,
                            bool transposeB )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Cuda >, "This function is specialized for CUDA device only." );

   cublasHandle_t handle;
   cublasCreate( &handle );

   IndexType m = transposeA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transposeB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transposeA ? matrix1.getRows() : matrix1.getColumns();

   cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
   cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

   IndexType lda = transposeA ? k : m;
   IndexType ldb = transposeB ? n : k;
   IndexType ldc = m;

   RealType alpha = 1.0;
   RealType beta = 0.0;

   if constexpr( std::is_same_v< RealType, float > ) {
      cublasSgemm( handle,
                   opA,
                   opB,
                   m,
                   n,
                   k,
                   &alpha,
                   matrix1.getValues().getData(),
                   lda,
                   matrix2.getValues().getData(),
                   ldb,
                   &beta,
                   resultMatrix.getValues().getData(),
                   ldc );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      cublasDgemm( handle,
                   opA,
                   opB,
                   m,
                   n,
                   k,
                   &alpha,
                   matrix1.getValues().getData(),
                   lda,
                   matrix2.getValues().getData(),
                   ldb,
                   &beta,
                   resultMatrix.getValues().getData(),
                   ldc );
   }

   cublasDestroy( handle );
}

}  // namespace TNL::Benchmarks::DenseMatrices

#endif  //__CUDACC__
