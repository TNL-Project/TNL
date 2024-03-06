#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__
   #ifdef HAVE_MAGMA

      #include <magma_v2.h>

template< typename DenseMatrix >
void
matrixMultiplicationMAGMA( const DenseMatrix& matrix1,
                           const DenseMatrix& matrix2,
                           DenseMatrix& resultMatrix,
                           bool transposeA,
                           bool transposeB )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;

   // Initialize MAGMA
   magma_init();

   // Create a MAGMA queue
   magma_queue_t queue = nullptr;
   magma_int_t dev = 0;
   magma_queue_create( dev, &queue );

   // Adjust dimensions based on transposition
   IndexType m = transposeA ? matrix1.getColumns() : matrix1.getRows();
   IndexType n = transposeB ? matrix2.getRows() : matrix2.getColumns();
   IndexType k = transposeA ? matrix1.getRows() : matrix1.getColumns();

   // Adjust leading dimensions based on transposition
   magma_int_t lda = transposeA ? k : m;
   magma_int_t ldb = transposeB ? n : k;
   magma_int_t ldc = m;

   // MAGMA GEMM parameters
   RealType alpha = 1.0;
   RealType beta = 0.0;
   magma_trans_t transA = transposeA ? MagmaTrans : MagmaNoTrans;
   magma_trans_t transB = transposeB ? MagmaTrans : MagmaNoTrans;

   // Perform the matrix multiplication using MAGMA
   if constexpr( std::is_same_v< RealType, float > ) {
      magma_sgemm( transA,
                   transB,
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
      magma_dgemm( transA,
                   transB,
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

   // Destroy the MAGMA queue
   magma_queue_destroy( queue );

   // Finalize MAGMA
   magma_finalize();
}

// Function to perform matrix transposition using MAGMA
template< typename RealType, typename DeviceType, typename IndexType >
void
denseMatrixTransposeMAGMA( const TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >& inputMatrix,
                           TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >& transposedMatrix )
{
   // Initialize MAGMA
   magma_init();

   // Create a MAGMA queue
   magma_queue_t queue = nullptr;
   magma_int_t dev = 0;
   magma_queue_create( dev, &queue );

   // Matrix dimensions
   int m = inputMatrix.getRows();
   int n = inputMatrix.getColumns();

   // Prepare the input and output data
   const RealType* d_input = inputMatrix.getValues().getData();
   RealType* d_transposed = transposedMatrix.getValues().getData();

   // Perform the matrix transposition using MAGMA
   if constexpr( std::is_same_v< RealType, float > ) {
      magmablas_stranspose( m, n, d_input, m, d_transposed, n, queue );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      magmablas_dtranspose( m, n, d_input, m, d_transposed, n, queue );
   }

   // Destroy the MAGMA queue
   magma_queue_destroy( queue );

   // Finalize MAGMA
   magma_finalize();
}

   #endif  // HAVE_MAGMA
#endif     // __CUDACC__
