// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseOperations.h>
#include <type_traits>

#ifdef __CUDACC__

   #include <cublas_v2.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationCuBLAS(
   const DenseMatrix& matrix1,
   const DenseMatrix& matrix2,
   DenseMatrix& resultMatrix,
   TNL::Matrices::TransposeState transposeA,
   TNL::Matrices::TransposeState transposeB )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::GPU >, "This function is specialized for GPU device only." );

   cublasHandle_t handle;
   cublasCreate( &handle );

   bool transA = transposeA == TNL::Matrices::TransposeState::Transpose;
   bool transB = transposeB == TNL::Matrices::TransposeState::Transpose;

   IndexType m = transA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transA ? matrix1.getRows() : matrix1.getColumns();

   cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
   cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

   IndexType lda = transA ? k : m;
   IndexType ldb = transB ? n : k;
   IndexType ldc = m;

   RealType alpha = 1.0;
   RealType beta = 0.0;

   if constexpr( std::is_same_v< RealType, float > ) {
      cublasSgemm(
         handle,
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
      cublasDgemm(
         handle,
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
