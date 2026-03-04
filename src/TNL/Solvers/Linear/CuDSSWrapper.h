// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_CUDSS
   #include <cudss.h>

   // cudss API error checking
   #define TNL_CUDSS_CHECK( err )                                                                               \
      {                                                                                                         \
         cudssStatus_t err_ = ( err );                                                                          \
         if( err_ != CUDSS_STATUS_SUCCESS ) {                                                                   \
            const std::string err_code = std::to_string( static_cast< int >( err_ ) );                          \
            const std::string message =                                                                         \
               "cudss error " + err_code + " at " + std::string( __FILE__ ) + ":" + std::to_string( __LINE__ ); \
            throw std::runtime_error( message );                                                                \
         }                                                                                                      \
      }
#endif

#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
class CuDSSWrapper : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_sparse_csr_matrix_v< Matrix >, "The CuDSSWrapper solver is available only for CSR matrices." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda >, "CuDSSWrapper is available only on CUDA" );
   static_assert( std::is_same_v< typename Matrix::RealType, float > || std::is_same_v< typename Matrix::RealType, double >,
                  "unsupported RealType" );
   static_assert( std::is_same_v< typename Matrix::IndexType, std::int32_t >
                     || std::is_same_v< typename Matrix::IndexType, std::uint32_t >
                     || std::is_same_v< typename Matrix::IndexType, std::int64_t >
                     || std::is_same_v< typename Matrix::IndexType, std::uint64_t >,
                  "unsupported IndexType" );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using MatrixPointer = typename Base::MatrixPointer;

   CuDSSWrapper()
   {
#ifdef HAVE_CUDSS
      TNL_CUDSS_CHECK( cudssCreate( &handle ) );
      TNL_CUDSS_CHECK( cudssConfigCreate( &solverConfig ) );
      TNL_CUDSS_CHECK( cudssDataCreate( handle, &solverData ) );

      if constexpr( std::is_same_v< RealType, float > )
         valueType = CUDA_R_32F;
      if constexpr( std::is_same_v< RealType, double > )
         valueType = CUDA_R_64F;

      if constexpr( std::is_same_v< IndexType, std::int32_t > )
         indexType = CUDA_R_32I;
      if constexpr( std::is_same_v< IndexType, std::uint32_t > )
         indexType = CUDA_R_32U;
      if constexpr( std::is_same_v< IndexType, std::int64_t > )
         indexType = CUDA_R_64I;
      if constexpr( std::is_same_v< IndexType, std::uint64_t > )
         indexType = CUDA_R_64U;
#endif
   }

   void
   setMatrix( const MatrixPointer& matrix ) override
   {
#ifdef HAVE_CUDSS
      LinearSolver< Matrix >::setMatrix( matrix );

      this->destroy();
      TNL_CUDSS_CHECK( cudssMatrixCreateCsr(  //
         &A,
         matrix->getRows(),
         matrix->getColumns(),
         matrix->getNonzeroElementsCount(),
         (void*) matrix->getSegments().getOffsets().getData(),
         nullptr,
         (void*) matrix->getColumnIndexes().getData(),
         (void*) matrix->getValues().getData(),
         indexType,
         valueType,
         CUDSS_MTYPE_GENERAL,
         CUDSS_MVIEW_FULL,
         CUDSS_BASE_ZERO ) );

      TNL_CUDSS_CHECK( cudssExecute( handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, nullptr, nullptr ) );

      TNL_CUDSS_CHECK( cudssExecute( handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, nullptr, nullptr ) );
#else
      throw std::runtime_error( "CuDSSWrapper was not built with CuDSS support" );
#endif
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef HAVE_CUDSS
      if( this->matrix->getColumns() != x.getSize() )
         throw std::invalid_argument( "CuDSSWrapper::solve: wrong size of the solution vector" );
      if( this->matrix->getColumns() != b.getSize() )
         throw std::invalid_argument( "CuDSSWrapper::solve: wrong size of the right hand side" );

      this->setResidue( NAN );
      if( A == nullptr ) {
         throw std::runtime_error( "CuDSSWrapper is not ready for solving." );
      }

      // create temporary matrices/vectors
      const IndexType size = this->matrix->getColumns();
      cudssMatrix_t _x = nullptr;
      cudssMatrix_t _b = nullptr;
      TNL_CUDSS_CHECK( cudssMatrixCreateDn( &_b, size, 1, size, (void*) b.getData(), valueType, CUDSS_LAYOUT_COL_MAJOR ) );
      TNL_CUDSS_CHECK( cudssMatrixCreateDn( &_x, size, 1, size, (void*) x.getData(), valueType, CUDSS_LAYOUT_COL_MAJOR ) );

      // solve the system
      TNL_CUDSS_CHECK( cudssExecute( handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, _x, _b ) );
      this->setResidue( 0 );

      // destroy temporary matrices/vectors
      TNL_CUDSS_CHECK( cudssMatrixDestroy( _x ) );
      TNL_CUDSS_CHECK( cudssMatrixDestroy( _b ) );

      return true;
#else
      throw std::runtime_error( "CuDSSWrapper was not built with CuDSS support" );
#endif
   }

   void
   destroy()
   {
#ifdef HAVE_CUDSS
      if( A != nullptr ) {
         TNL_CUDSS_CHECK( cudssMatrixDestroy( A ) );
         A = nullptr;
      }
#endif
   }

   ~CuDSSWrapper()
   {
#ifdef HAVE_CUDSS
      this->destroy();
      try {
         TNL_CUDSS_CHECK( cudssConfigDestroy( solverConfig ) );
         TNL_CUDSS_CHECK( cudssDataDestroy( handle, solverData ) );
         TNL_CUDSS_CHECK( cudssDestroy( handle ) );
      }
      catch( std::exception& e ) {
         std::cerr << "Error in CuDSSWrapper destructor: " << e.what() << "\n";
      }
#endif
   }

protected:
#ifdef HAVE_CUDSS
   cudssHandle_t handle;
   cudaDataType_t indexType;
   cudaDataType_t valueType;
   cudssConfig_t solverConfig;
   cudssData_t solverData;
   cudssMatrix_t A = nullptr;
#endif
};

}  // namespace TNL::Solvers::Linear
