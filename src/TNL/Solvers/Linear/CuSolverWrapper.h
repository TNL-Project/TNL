// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __CUDACC__
   #include <cusolverDn.h>

   // cudss API error checking
   #define TNL_CUSOLVER_CHECK( err )                                                                               \
      {                                                                                                            \
         cusolverStatus_t err_ = ( err );                                                                          \
         if( err_ != CUSOLVER_STATUS_SUCCESS ) {                                                                   \
            const std::string err_code = std::to_string( static_cast< int >( err_ ) );                             \
            const std::string message =                                                                            \
               "cusolver error " + err_code + " at " + std::string( __FILE__ ) + ":" + std::to_string( __LINE__ ); \
            throw std::runtime_error( message );                                                                   \
         }                                                                                                         \
      }
#endif

#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
class CuSolverWrapper : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_dense_matrix_v< Matrix >, "The CuSolverWrapper solver is available only for dense matrices." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda >, "CuSolverWrapper is available only on CUDA" );
   static_assert(
      std::is_same_v< typename Matrix::RealType, float > || std::is_same_v< typename Matrix::RealType, double >,
      "unsupported RealType" );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   CuSolverWrapper()
   {
#ifdef __CUDACC__
      TNL_CUSOLVER_CHECK( cusolverDnCreate( &handle ) );
#endif
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef __CUDACC__
      if( this->matrix->getColumns() != x.getSize() )
         throw std::invalid_argument( "CuSolverWrapper::solve: wrong size of the solution vector" );
      if( this->matrix->getColumns() != b.getSize() )
         throw std::invalid_argument( "CuSolverWrapper::solve: wrong size of the right hand side" );

      this->setResidue( NAN );

      const IndexType n = this->matrix->getRows();
      RealType* data = const_cast< RealType* >( this->matrix->getValues().getData() );

      int lwork = 0;
      if constexpr( std::is_same_v< RealType, float > ) {
         TNL_CUSOLVER_CHECK( cusolverDnSgetrf_bufferSize( handle, n, n, data, n, &lwork ) );
      }
      else if constexpr( std::is_same_v< RealType, double > ) {
         TNL_CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize( handle, n, n, data, n, &lwork ) );
      }

      Containers::Vector< RealType, Devices::Cuda > d_work( lwork );
      Containers::Vector< int, Devices::Cuda > d_pivots( n );
      Containers::Vector< int, Devices::Cuda > d_info( 1 );

      if constexpr( std::is_same_v< RealType, float > ) {
         // LU factorization
         TNL_CUSOLVER_CHECK(
            cusolverDnSgetrf( handle, n, n, data, n, d_work.getData(), d_pivots.getData(), d_info.getData() ) );

         // Solve Ax = b
         TNL_CUSOLVER_CHECK(
            cusolverDnSgetrs( handle, CUBLAS_OP_N, n, 1, data, n, d_pivots.getData(), x.getData(), n, d_info.getData() ) );
      }
      else if constexpr( std::is_same_v< RealType, double > ) {
         // LU factorization
         TNL_CUSOLVER_CHECK(
            cusolverDnDgetrf( handle, n, n, data, n, d_work.getData(), d_pivots.getData(), d_info.getData() ) );

         // Solve Ax = b
         TNL_CUSOLVER_CHECK(
            cusolverDnDgetrs( handle, CUBLAS_OP_N, n, 1, data, n, d_pivots.getData(), x.getData(), n, d_info.getData() ) );
      }

      this->setResidue( 0 );
      return true;
#else
      throw std::runtime_error( "CuSolverWrapper was not built with CUDA support." );
#endif
   }

   ~CuSolverWrapper()
   {
#ifdef __CUDACC__
      try {
         cusolverDnDestroy( handle );
      }
      catch( std::exception& e ) {
         std::cerr << "Error in CuSolverWrapper destructor: " << e.what() << "\n";
      }
#endif
   }

protected:
#ifdef __CUDACC__
   cusolverDnHandle_t handle;
#endif
};

}  // namespace TNL::Solvers::Linear
