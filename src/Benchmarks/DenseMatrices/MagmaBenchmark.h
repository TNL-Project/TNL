// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseOperations.h>
#include <type_traits>

#ifdef __CUDACC__
   #ifdef HAVE_MAGMA

      #include <magma_v2.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationMAGMA(
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

   magma_init();

   magma_queue_t queue = nullptr;
   magma_int_t dev = 0;
   magma_queue_create( dev, &queue );

   bool transA = transposeA == TNL::Matrices::TransposeState::Transpose;
   bool transB = transposeB == TNL::Matrices::TransposeState::Transpose;

   IndexType m = transA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transA ? matrix1.getRows() : matrix1.getColumns();

   magma_int_t lda = transA ? k : m;
   magma_int_t ldb = transB ? n : k;
   magma_int_t ldc = m;

   RealType alpha = 1.0;
   RealType beta = 0.0;
   magma_trans_t magTransA = transA ? MagmaTrans : MagmaNoTrans;
   magma_trans_t magTransB = transB ? MagmaTrans : MagmaNoTrans;

   if constexpr( std::is_same_v< RealType, float > ) {
      magma_sgemm(
         magTransA,
         magTransB,
         m,
         n,
         k,
         alpha,
         matrix1.getValues().getData(),
         lda,
         matrix2.getValues().getData(),
         ldb,
         beta,
         resultMatrix.getValues().getData(),
         ldc,
         queue );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      magma_dgemm(
         magTransA,
         magTransB,
         m,
         n,
         k,
         alpha,
         matrix1.getValues().getData(),
         lda,
         matrix2.getValues().getData(),
         ldb,
         beta,
         resultMatrix.getValues().getData(),
         ldc,
         queue );
   }

   magma_queue_destroy( queue );

   magma_finalize();
}

template< typename RealType, typename DeviceType, typename IndexType >
void
denseMatrixTransposeMAGMA(
   const TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >& inputMatrix,
   TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >& transposedMatrix )
{
   magma_init();

   magma_queue_t queue = nullptr;
   magma_int_t dev = 0;
   magma_queue_create( dev, &queue );

   int m = inputMatrix.getRows();
   int n = inputMatrix.getColumns();

   const RealType* d_input = inputMatrix.getValues().getData();
   RealType* d_transposed = transposedMatrix.getValues().getData();

   if constexpr( std::is_same_v< RealType, float > ) {
      magmablas_stranspose( m, n, d_input, m, d_transposed, n, queue );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      magmablas_dtranspose( m, n, d_input, m, d_transposed, n, queue );
   }

   magma_queue_destroy( queue );

   magma_finalize();
}

}  // namespace TNL::Benchmarks::DenseMatrices

   #endif  // HAVE_MAGMA
#endif     // __CUDACC__
