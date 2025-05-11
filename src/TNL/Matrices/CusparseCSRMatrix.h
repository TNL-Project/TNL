#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SparseMatrixBase.h>
#ifdef __CUDACC__
   #include <cusparse.h>
#endif

#include <cstddef>
//#include <stdexcept>

#ifdef __CUDACC__
   #define CHECK_CUSPARSE( func )                                                                                          \
      {                                                                                                                    \
         cusparseStatus_t status = ( func );                                                                               \
         if( status != CUSPARSE_STATUS_SUCCESS ) {                                                                         \
            std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: " << cusparseGetErrorString( status ) \
                      << " " << status << std::endl;                                                                       \
            throw std::runtime_error( cusparseGetErrorString( status ) );                                                  \
         }                                                                                                                 \
      }
#endif

namespace TNL::Matrices {

template< typename CSRMatrix >
class CusparseCSRMatrix
{
public:
   using RealType = typename CSRMatrix::RealType;
   using DeviceType = TNL::Devices::Cuda;
   using IndexType = typename CSRMatrix::IndexType;
   using MatrixType = CSRMatrix;

   static_assert( std::is_same_v< typename MatrixType::IndexType, int >
                     || std::is_same_v< typename MatrixType::IndexType, long >,
                  "CusparseCSRMatrix can only be used with int and long int index type" );
   static_assert( std::is_same_v< typename MatrixType::RealType, double >
                     || std::is_same_v< typename MatrixType::RealType, float >,
                  "CusparseCSRMatrix can only be used with double and float real type" );

   CusparseCSRMatrix()
   {
#ifdef __CUDACC__
      if( std::is_same_v< typename MatrixType::IndexType, int > ) {
         cusparseIndexType = CUSPARSE_INDEX_32I;
      }
      else if( std::is_same_v< typename MatrixType::IndexType, long > ) {
         cusparseIndexType = CUSPARSE_INDEX_64I;
      }
      else {
         throw std::runtime_error( "CusparseCSRMatrix: Unsupported index type" );
      }
      if( std::is_same_v< typename MatrixType::RealType, float > ) {
         cusparseValueType = CUDA_R_32F;
      }
      else if( std::is_same_v< typename MatrixType::RealType, double > ) {
         cusparseValueType = CUDA_R_64F;
      }
      else if( std::is_same_v< typename MatrixType::RealType, std::complex< float > > ) {
         cusparseValueType = CUDA_C_32F;
      }
      else if( std::is_same_v< typename MatrixType::RealType, std::complex< double > > ) {
         cusparseValueType = CUDA_C_64F;
      }
      else {
         throw std::runtime_error( "CusparseCSRMatrix: Unsupported real type" );
      }
#endif
   }

   IndexType
   getRows() const
   {
      return matrix->getRows();
   }

   IndexType
   getColumns() const
   {
      return matrix->getColumns();
   }

   IndexType
   getNumberOfMatrixElements() const
   {
      return matrix->getAllocatedElementsCount();
   }

   void
   init( const MatrixType& matrix )
   {
      if( ! std::is_same_v< typename MatrixType::DeviceType, DeviceType > )
         throw std::runtime_error( "CusparseCSRMatrix can only be used with Cuda matrices" );

#ifdef __CUDACC__
      this->matrix = &matrix;
   #if CUDART_VERSION >= 11000
      CHECK_CUSPARSE( cusparseCreateCsr( &this->matA,                   // cusparseSpMatDescr_t* spMatDescr,
                                         matrix.getRows(),              // int64_t               rows,
                                         matrix.getColumns(),           // int64_t               cols,
                                         matrix.getValues().getSize(),  // int64_t               nnz,
                                         (void*) matrix.getSegments().getOffsets().getData(),  // void* csrRowOffsets,
                                         (void*) matrix.getColumnIndexes().getData(),  // void*                 csrColInd,
                                         (void*) matrix.getValues().getData(),         // void*                 csrValues,
                                         cusparseIndexType,         // cusparseIndexType_t   csrRowOffsetsType,
                                         cusparseIndexType,         // cusparseIndexType_t   csrColIndType,
                                         CUSPARSE_INDEX_BASE_ZERO,  // cusparseIndexBase_t   idxBase,
                                         cusparseValueType ) );     // cudaDataType          valueType)
   #else
      CHECK_CUSPARSE( cusparseCreateMatDescr( &this->matrixDescriptor ) );
   #endif
#endif
   }

   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector, OutVector& outVector ) const
   {
#ifdef __CUDACC__
      TNL_ASSERT_TRUE( this->matrix, "matrix was not initialized" );

      cusparseDnVecDescr_t vecX, vecY;
      CHECK_CUSPARSE( cusparseCreateDnVec( &vecX, this->matrix->getColumns(), (void*) inVector.getData(), cusparseValueType ) );
      CHECK_CUSPARSE( cusparseCreateDnVec( &vecY, this->matrix->getRows(), (void*) outVector.getData(), cusparseValueType ) );

   #if CUDART_VERSION >= 11000

      RealType alpha = 1.0;
      RealType beta = 0.0;
      size_t buffer_size;

      cusparseHandle_t cusparseHandle;
      CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ) );
      CHECK_CUSPARSE( cusparseSpMV_bufferSize( cusparseHandle,                    // cusparseHandle_t     handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,  // cusparseOperation_t  opA,
                                               &alpha,                            // const void*          alpha,
                                               this->matA,                        // cusparseSpMatDescr_t matA,
                                               vecX,                              // cusparseDnVecDescr_t vecX,
                                               &alpha,                            // const void*          beta,
                                               vecY,                              // cusparseDnVecDescr_t vecY,
                                               cusparseValueType,                 // cudaDataType         computeType,
                                               CUSPARSE_SPMV_ALG_DEFAULT,         // cusparseSpMVAlg_t    alg,
                                               &buffer_size ) );                  // size_t*              bufferSize)
      this->buffer.setSize( buffer_size );
      CHECK_CUSPARSE( cusparseSpMV( cusparseHandle,                      // cusparseHandle_t     handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,    // cusparseOperation_t  opA,
                                    &alpha,                              // const void*          alpha,
                                    this->matA,                          // cusparseSpMatDescr_t matA,
                                    vecX,                                // cusparseDnVecDescr_t vecX,
                                    &beta,                               // const void*          beta,
                                    vecY,                                // cusparseDnVecDescr_t vecY,
                                    cusparseValueType,                   // cudaDataType         computeType,
                                    CUSPARSE_SPMV_ALG_DEFAULT,           // cusparseSpMVAlg_t    alg,
                                    (void*) this->buffer.getData() ) );  // void*                externalBuffer)
   #else

      RealType a = 1.0;
      RealType b = 0.0;
      RealType* alpha = &a;
      RealType* beta = &b;
      CHECK_CUSPARSE( cusparseDcsrmv( cusparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      this->matrix->getRows(),
                                      this->matrix->getColumns(),
                                      this->matrix->getValues().getSize(),
                                      alpha,
                                      this->matrixDescriptor,
                                      this->matrix->getValues().getData(),
                                      this->matrix->getSegments().getOffsets().getData(),
                                      this->matrix->getColumnIndexes().getData(),
                                      inVector.getData(),
                                      beta,
                                      outVector.getData() ) );

      CHECK_CUSPARSE( cusparseDestroyDnMat( vecX ) );
      CHECK_CUSPARSE( cusparseDestroyDnMat( vecY ) );
      CHECK_CUSPARSE( cusparseDestroy( cusparseHandle ) );
   #endif
#endif
   }

   template< typename InVector, typename OutVector >
   void
   vectorsProduct( const InVector& inVector1, const InVector& inVector2, OutVector& outVector1, OutVector& outVector2 ) const
   {
#ifdef __CUDACC__
   #if CUDART_VERSION >= 11000
      TNL_ASSERT_TRUE( this->matrix, "matrix was not initialized" );
      TNL_ASSERT_EQ( inVector1.getSize(), this->matrix->getColumns(), "inVector1 size does not match matrix columns" );
      TNL_ASSERT_EQ( inVector2.getSize(), this->matrix->getColumns(), "inVector2 size does not match matrix columns" );
      TNL_ASSERT_EQ( outVector1.getSize(), this->matrix->getRows(), "outVector1 size does not match matrix rows" );
      TNL_ASSERT_EQ( outVector2.getSize(), this->matrix->getRows(), "outVector2 size does not match matrix rows" );

      const auto rows = this->matrix->getRows();
      const auto columns = this->matrix->getColumns();

      Containers::Vector< RealType, DeviceType, IndexType > inVectors( 2 * columns );
      inVectors.getView( 0, columns ) = inVector1;
      inVectors.getView( columns, 2 * columns ) = inVector2;
      Containers::Vector< RealType, DeviceType, IndexType > outVectors( 2 * rows );

      cusparseDnMatDescr_t matX, matY;
      CHECK_CUSPARSE(
         cusparseCreateDnMat( &matX, columns, 2, columns, inVectors.getData(), cusparseValueType, CUSPARSE_ORDER_COL ) );
      CHECK_CUSPARSE(
         cusparseCreateDnMat( &matY, rows, 2, rows, outVectors.getData(), cusparseValueType, CUSPARSE_ORDER_COL ) );

      RealType alpha = 1.0, beta = 0.0;
      size_t bufferSize = 0;

      cusparseHandle_t cusparseHandle;
      CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ) );

      CHECK_CUSPARSE( cusparseSpMM_bufferSize( cusparseHandle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matX,
                                               &beta,
                                               matY,
                                               cusparseValueType,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &bufferSize ) );
      this->buffer.setSize( bufferSize );

      CHECK_CUSPARSE( cusparseSpMM( cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matX,
                                    &beta,
                                    matY,
                                    cusparseValueType,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    this->buffer.getData() ) );

      outVector1 = outVectors.getView( 0, rows );
      outVector2 = outVectors.getView( rows, 2 * rows );

      CHECK_CUSPARSE( cusparseDestroyDnMat( matX ) );
      CHECK_CUSPARSE( cusparseDestroyDnMat( matY ) );
      CHECK_CUSPARSE( cusparseDestroy( cusparseHandle ) );
   #else
      throw std::runtime_error( "Multiplication of matrix and two vectors is supported only for CUDA 11 and newer." );
   #endif
#endif
   }

protected:
   const MatrixType* matrix = nullptr;
#ifdef __CUDACC__
   #if CUDART_VERSION < 11000
   cusparseMatDescr_t matrixDescriptor;
   #else
   cusparseSpMatDescr_t matA;
   mutable TNL::Containers::Array< std::byte, TNL::Devices::Cuda > buffer;
   cusparseIndexType_t cusparseIndexType;
   cudaDataType cusparseValueType;
   #endif
#endif
};

}  //namespace TNL::Matrices
