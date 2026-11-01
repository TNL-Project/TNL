// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __CUDACC__ ) && defined( HAVE_CUSPARSE )

   #include <stdexcept>
   #include <string>
   #include <type_traits>

   #include <cusparse.h>

   #include <TNL/Containers/Array.h>
   #include <TNL/Devices/GPU.h>

   #include "ReferenceKernels.h"

namespace TNL::Benchmarks::SpMM {

inline void
checkCuSparse( cusparseStatus_t status, const char* call )
{
   if( status != CUSPARSE_STATUS_SUCCESS )
      throw std::runtime_error( std::string( call ) + " failed: " + cusparseGetErrorString( status ) );
}

   #define TNL_CUSPARSE_CALL( call ) checkCuSparse( call, #call )

/**
 * \brief Wrapper of `cusparseSpMM` for a CSR matrix and dense matrices.
 *
 * The descriptors, the external buffer and the optional preprocessing are
 * set up in the constructor, so that \ref multiply measures only the
 * multiplication itself.
 */
template< typename Real, typename Index >
class CuSparseSpMM
{
public:
   CuSparseSpMM(
      const CSRArrays< Real, Index >& A,
      const DenseArray< Real, Index >& B,
      DenseArray< Real, Index >& C,
      bool rowMajor,
      cusparseSpMMAlg_t algorithm )
   : algorithm( algorithm )
   {
      static_assert(
         std::is_same_v< Index, int > || std::is_same_v< Index, long long int >,
         "cuSPARSE supports only 32-bit and 64-bit indices." );
      constexpr cusparseIndexType_t indexType = std::is_same_v< Index, int > ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
      const cusparseOrder_t order = rowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

      TNL_CUSPARSE_CALL( cusparseCreate( &handle ) );
      TNL_CUSPARSE_CALL( cusparseCreateCsr(
         &matA,
         A.rows,
         A.columns,
         nnz( A ),
         const_cast< Index* >( A.rowPointers ),
         const_cast< Index* >( A.columnIndexes ),
         const_cast< Real* >( A.values ),
         indexType,
         indexType,
         CUSPARSE_INDEX_BASE_ZERO,
         dataType() ) );
      TNL_CUSPARSE_CALL( cusparseCreateDnMat( &matB, B.rows, B.columns, B.leadingDimension, B.data, dataType(), order ) );
      TNL_CUSPARSE_CALL( cusparseCreateDnMat( &matC, C.rows, C.columns, C.leadingDimension, C.data, dataType(), order ) );

      std::size_t bufferSize = 0;
      TNL_CUSPARSE_CALL( cusparseSpMM_bufferSize(
         handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         matA,
         matB,
         &beta,
         matC,
         dataType(),
         algorithm,
         &bufferSize ) );
      buffer.setSize( std::max( bufferSize, std::size_t( 1 ) ) );
      if( algorithm == CUSPARSE_SPMM_CSR_ALG3 )
         TNL_CUSPARSE_CALL( cusparseSpMM_preprocess(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            &beta,
            matC,
            dataType(),
            algorithm,
            buffer.getData() ) );
   }

   CuSparseSpMM( const CuSparseSpMM& ) = delete;
   CuSparseSpMM&
   operator=( const CuSparseSpMM& ) = delete;

   ~CuSparseSpMM()
   {
      cusparseDestroySpMat( matA );
      cusparseDestroyDnMat( matB );
      cusparseDestroyDnMat( matC );
      cusparseDestroy( handle );
   }

   //! \brief Computes C = A * B and waits for the result.
   void
   multiply()
   {
      TNL_CUSPARSE_CALL( cusparseSpMM(
         handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         matA,
         matB,
         &beta,
         matC,
         dataType(),
         algorithm,
         buffer.getData() ) );
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );
   }

private:
   static constexpr cudaDataType
   dataType()
   {
      static_assert( std::is_same_v< Real, float > || std::is_same_v< Real, double >, "Only float and double are supported." );
      return std::is_same_v< Real, float > ? CUDA_R_32F : CUDA_R_64F;
   }

   static Index
   nnz( const CSRArrays< Real, Index >& A )
   {
      Index result = 0;
      TNL_BACKEND_SAFE_CALL( cudaMemcpy( &result, A.rowPointers + A.rows, sizeof( Index ), cudaMemcpyDeviceToHost ) );
      return result;
   }

   cusparseSpMMAlg_t algorithm;
   cusparseHandle_t handle = nullptr;
   cusparseSpMatDescr_t matA = nullptr;
   cusparseDnMatDescr_t matB = nullptr;
   cusparseDnMatDescr_t matC = nullptr;
   Containers::Array< char, Devices::GPU, std::size_t > buffer;
   Real alpha = 1;
   Real beta = 0;
};

   #undef TNL_CUSPARSE_CALL

}  // namespace TNL::Benchmarks::SpMM

#endif
