#pragma once

#if defined( __HIP__ )

   #include <hipblas/hipblas.h>
// Function to perform matrix multiplication using HIPBLAS
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

   hipblasHandle_t handle;
   hipblasCreate( &handle );

   // Adjust matrix dimensions based on transposition
   IndexType m = transposeA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transposeB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transposeA ? matrix1.getRows() : matrix1.getColumns();

   // HIPBLAS operation flags for transposition
   hipblasOperation_t opA = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
   hipblasOperation_t opB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

   // Adjust leading dimensions based on transposition
   IndexType lda = transposeA ? k : m;
   IndexType ldb = transposeB ? n : k;
   IndexType ldc = m;

   RealType alpha = 1.0;
   RealType beta = 0.0;

   // Perform the matrix multiplication using HIPBLAS
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

#endif  // (__HIPP__)
