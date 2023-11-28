#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__

#include <cublas_v2.h>

// Function to perform matrix multiplication using CuBLAS
template <typename RealType, typename DeviceType, typename IndexType>
void matrixMultiplicationCuBLAS(const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix1,
                                const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix2,
                                TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& resultMatrix) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Ensure proper dimensions for matrix multiplication
    int m = matrix1.getRows();    // number of rows of matrix 1 and result
    int k = matrix1.getColumns(); // number of columns of matrix 1 and rows of matrix 2
    int n = matrix2.getColumns(); // number of columns of matrix 2 and result

    // Call CuBLAS function with the matrix data
    if constexpr( std::is_same_v< RealType, float > ) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    matrix2.getValues().getData(), n,
                    matrix1.getValues().getData(), k, &beta,
                    resultMatrix.getValues().getData(), n);
    } else if constexpr( std::is_same_v< RealType, double > ) {
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    matrix2.getValues().getData(), n,
                    matrix1.getValues().getData(), k, &beta,
                    resultMatrix.getValues().getData(), n);
    }

}
#endif
