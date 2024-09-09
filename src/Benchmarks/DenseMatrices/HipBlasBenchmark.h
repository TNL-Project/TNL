// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __HIP__ )

   #include <hipblas/hipblas.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationHIPBLAS( const DenseMatrix& matrix1,
                             const DenseMatrix& matrix2,
                             DenseMatrix& resultMatrix,
                             bool transposeA,
                             bool transposeB )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Hip >, "This function is specialized for Hip device only." );

   hipblasHandle_t handle;
   hipblasCreate( &handle );

   IndexType m = transposeA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transposeB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transposeA ? matrix1.getRows() : matrix1.getColumns();

   hipblasOperation_t opA = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
   hipblasOperation_t opB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

   IndexType lda = transposeA ? k : m;
   IndexType ldb = transposeB ? n : k;
   IndexType ldc = m;

   RealType alpha = 1.0;
   RealType beta = 0.0;

   if constexpr( std::is_same_v< RealType, float > ) {
      hipblasSgemm( handle,
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
      hipblasDgemm( handle,
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

   hipblasDestroy( handle );
}

}  // namespace TNL::Benchmarks::DenseMatrices

#endif  // (__HIPP__)
