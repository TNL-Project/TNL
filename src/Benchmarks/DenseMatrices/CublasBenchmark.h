#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__

#include <cublas_v2.h>

// Function to perform matrix multiplication using CuBLAS
template <typename DenseMatrix>
void matrixMultiplicationCuBLAS(const DenseMatrix& matrix1,
                                const DenseMatrix& matrix2,
                                DenseMatrix& resultMatrix) {

   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;

   cublasHandle_t handle;
   cublasCreate(&handle);

   // Matrix dimensions
   IndexType m = matrix1.getRows();    // number of rows in matrix1 (and result)
   IndexType n = matrix2.getColumns(); // number of columns in matrix2 (and result)
   IndexType k = matrix1.getColumns(); // number of columns in matrix1 (and rows in matrix2)

   // Setting up the parameters for cuBLAS
   RealType alpha = 1.0;
   RealType beta = 0.0;

   // Leading dimensions based on column-major format
   IndexType lda = m; // Leading dimension of matrix1
   IndexType ldb = k; // Leading dimension of matrix2
   IndexType ldc = m; // Leading dimension of resultMatrix

   // Perform the matrix multiplication using cuBLAS
   if constexpr(std::is_same_v<RealType, float>) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  matrix1.getValues().getData(), lda,
                  matrix2.getValues().getData(), ldb, &beta,
                  resultMatrix.getValues().getData(), ldc);
   } else if constexpr(std::is_same_v<RealType, double>) {
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  matrix1.getValues().getData(), lda,
                  matrix2.getValues().getData(), ldb, &beta,
                  resultMatrix.getValues().getData(), ldc);
   }
   cublasDestroy(handle);
}

#endif //__CUDACC__
