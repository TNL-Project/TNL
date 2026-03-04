// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __CUDACC__
   #include <cusolverDn.h>
#endif

#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
class CuSolverWrapper : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_dense_matrix_v< Matrix >, "The CuSolverWrapper solver is available only for dense matrices." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda >, "CuSolverWrapper is available only on CUDA" );
   static_assert( std::is_same_v< typename Matrix::RealType, float > || std::is_same_v< typename Matrix::RealType, double >,
                  "unsupported RealType" );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef __CUDACC__
      TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

      this->resetIterations();
      this->setResidue( NAN );

      const IndexType n = this->matrix->getRows();

      cusolverDnHandle_t handle;
      cusolverStatus_t status = cusolverDnCreate( &handle );
      if( status != CUSOLVER_STATUS_SUCCESS ) {
         cusolverDnDestroy( handle );
         throw std::runtime_error( "cusolverDnCreate failed: " + std::to_string( status ) );
      }

      int lwork = 0;
      if constexpr( std::is_same_v< RealType, float > )
         status =
            cusolverDnSgetrf_bufferSize( handle, n, n, const_cast< float* >( this->matrix->getValues().getData() ), n, &lwork );
      else if constexpr( std::is_same_v< RealType, double > )
         status = cusolverDnDgetrf_bufferSize(
            handle, n, n, const_cast< double* >( this->matrix->getValues().getData() ), n, &lwork );

      Containers::Vector< RealType, Devices::Cuda > d_work( lwork );
      Containers::Vector< int, Devices::Cuda > d_pivots( n );
      Containers::Vector< int, Devices::Cuda > d_info( 1 );

      // LU factorization
      if constexpr( std::is_same_v< RealType, float > )
         status = cusolverDnSgetrf( handle,
                                    n,
                                    n,
                                    const_cast< float* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_work.getData(),
                                    d_pivots.getData(),
                                    d_info.getData() );
      else if constexpr( std::is_same_v< RealType, double > )
         status = cusolverDnDgetrf( handle,
                                    n,
                                    n,
                                    const_cast< double* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_work.getData(),
                                    d_pivots.getData(),
                                    d_info.getData() );

      // Solve Ax = b
      if constexpr( std::is_same_v< RealType, float > )
         status = cusolverDnSgetrs( handle,
                                    CUBLAS_OP_N,
                                    n,
                                    1,
                                    const_cast< float* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_pivots.getData(),
                                    x.getData(),
                                    n,
                                    d_info.getData() );
      else if constexpr( std::is_same_v< RealType, double > )
         status = cusolverDnDgetrs( handle,
                                    CUBLAS_OP_N,
                                    n,
                                    1,
                                    const_cast< double* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_pivots.getData(),
                                    x.getData(),
                                    n,
                                    d_info.getData() );

      cusolverDnDestroy( handle );
      this->setResidue( 0 );
      return true;
#else
      throw std::runtime_error( "CuSolverWrapper was not built with CUDA support." );
#endif
   }
};

}  // namespace TNL::Solvers::Linear
