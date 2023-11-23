#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixBase.h>
#include <type_traits>

#ifdef HAVE_BLAS

#include <cblas.h>

// Function to perform matrix multiplication using BLAS
template <typename RealType, typename DeviceType, typename IndexType>
void matrixMultiplicationBLAS(const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix1,
                              const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix2,
                              TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& resultMatrix) {

    // Ensure proper dimensions for matrix multiplication
    int n = matrix2.getColumns();
    int k = matrix1.getColumns();
    int m = matrix1.getRows();

    constexpr auto organization = matrix1.getOrganization();

    // Call BLAS function with the matrix data
if constexpr (std::is_same_v<RealType, float>) {
    cblas_sgemm(organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
            CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
            matrix1.getValues().getData(), k,
            matrix2.getValues().getData(), n, 0.0f,
            resultMatrix.getValues().getData(),
            organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m);
} else if constexpr (std::is_same_v<RealType, double>) {
    cblas_dgemm(organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
            CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
            matrix1.getValues().getData(), k,
            matrix2.getValues().getData(), n, 0.0,
            resultMatrix.getValues().getData(),
            organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m);
}
}

#endif
