// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __HIP__ ) && defined( HAVE_HIPSPARSE )

   #include <stdexcept>
   #include <string>
   #include <type_traits>

   #include <hipsparse/hipsparse.h>

   #include <TNL/Containers/Array.h>
   #include <TNL/Devices/GPU.h>

   #include "ReferenceKernels.h"

namespace TNL::Benchmarks::SpMM {

inline void
checkHipSparse( hipsparseStatus_t status, const char* call )
{
   if( status != HIPSPARSE_STATUS_SUCCESS )
      throw std::runtime_error( std::string( call ) + " failed with status " + std::to_string( static_cast< int >( status ) ) );
}

   #define TNL_HIPSPARSE_CALL( call ) checkHipSparse( call, #call )

/**
 * \brief Wrapper of `hipsparseSpMM` for a CSR matrix and dense matrices.
 *
 * The descriptors, the external buffer and the optional preprocessing are
 * set up in the constructor, so that \ref multiply measures only the
 * multiplication itself.
 */
template< typename Real, typename Index >
class HipSparseSpMM
{
public:
   HipSparseSpMM(
      const CSRArrays< Real, Index >& A,
      const DenseArray< Real, Index >& B,
      DenseArray< Real, Index >& C,
      bool rowMajor,
      hipsparseSpMMAlg_t algorithm )
   : algorithm( algorithm )
   {
      static_assert(
         std::is_same_v< Index, int > || std::is_same_v< Index, long long int >,
         "hipSPARSE supports only 32-bit and 64-bit indices." );
      constexpr hipsparseIndexType_t indexType = std::is_same_v< Index, int > ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
      const hipsparseOrder_t order = rowMajor ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;

      TNL_HIPSPARSE_CALL( hipsparseCreate( &handle ) );
      TNL_HIPSPARSE_CALL( hipsparseCreateCsr(
         &matA,
         A.rows,
         A.columns,
         nnz( A ),
         const_cast< Index* >( A.rowPointers ),
         const_cast< Index* >( A.columnIndexes ),
         const_cast< Real* >( A.values ),
         indexType,
         indexType,
         HIPSPARSE_INDEX_BASE_ZERO,
         dataType() ) );
      TNL_HIPSPARSE_CALL( hipsparseCreateDnMat( &matB, B.rows, B.columns, B.leadingDimension, B.data, dataType(), order ) );
      TNL_HIPSPARSE_CALL( hipsparseCreateDnMat( &matC, C.rows, C.columns, C.leadingDimension, C.data, dataType(), order ) );

      std::size_t bufferSize = 0;
      TNL_HIPSPARSE_CALL( hipsparseSpMM_bufferSize(
         handle,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         matA,
         matB,
         &beta,
         matC,
         dataType(),
         algorithm,
         &bufferSize ) );
      buffer.setSize( std::max( bufferSize, std::size_t( 1 ) ) );
      TNL_HIPSPARSE_CALL( hipsparseSpMM_preprocess(
         handle,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         matA,
         matB,
         &beta,
         matC,
         dataType(),
         algorithm,
         buffer.getData() ) );
   }

   HipSparseSpMM( const HipSparseSpMM& ) = delete;
   HipSparseSpMM&
   operator=( const HipSparseSpMM& ) = delete;

   ~HipSparseSpMM()
   {
      hipsparseDestroySpMat( matA );
      hipsparseDestroyDnMat( matB );
      hipsparseDestroyDnMat( matC );
      hipsparseDestroy( handle );
   }

   //! \brief Computes C = A * B and waits for the result.
   void
   multiply()
   {
      TNL_HIPSPARSE_CALL( hipsparseSpMM(
         handle,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         HIPSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         matA,
         matB,
         &beta,
         matC,
         dataType(),
         algorithm,
         buffer.getData() ) );
      TNL_BACKEND_SAFE_CALL( hipDeviceSynchronize() );
   }

private:
   static constexpr hipDataType
   dataType()
   {
      static_assert( std::is_same_v< Real, float > || std::is_same_v< Real, double >, "Only float and double are supported." );
      return std::is_same_v< Real, float > ? HIP_R_32F : HIP_R_64F;
   }

   static Index
   nnz( const CSRArrays< Real, Index >& A )
   {
      Index result = 0;
      TNL_BACKEND_SAFE_CALL( hipMemcpy( &result, A.rowPointers + A.rows, sizeof( Index ), hipMemcpyDeviceToHost ) );
      return result;
   }

   hipsparseSpMMAlg_t algorithm;
   hipsparseHandle_t handle = nullptr;
   hipsparseSpMatDescr_t matA = nullptr;
   hipsparseDnMatDescr_t matB = nullptr;
   hipsparseDnMatDescr_t matC = nullptr;
   Containers::Array< char, Devices::GPU, std::size_t > buffer;
   Real alpha = 1;
   Real beta = 0;
};

   #undef TNL_HIPSPARSE_CALL

}  // namespace TNL::Benchmarks::SpMM

#endif
