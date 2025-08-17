// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __CUDACC__
   #include <cusolverSp.h>
   #include <cusolverDn.h>
#endif

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {

namespace Solvers::Linear {

template< typename Matrix >
class CuSolverWrapper : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_sparse_csr_matrix_v< Matrix > || Matrices::is_dense_matrix_v< Matrix >,
                  "The CuSolverWrapper solver is available only for CSR and dense matrices." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda >, "CuSolverWrapper is available only on CUDA" );
   static_assert( std::is_same_v< typename Matrix::RealType, float > || std::is_same_v< typename Matrix::RealType, double >,
                  "unsupported RealType" );
   static_assert( Matrices::is_dense_matrix_v< Matrix > || std::is_same_v< typename Matrix::IndexType, int >,
                  "unsupported IndexType" );

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
      if constexpr( Matrices::is_sparse_csr_matrix_v< Matrix > )
         return solveSparse( b, x );
      else  // is_dense_matrix_v< Matrix > must be true
         return solveDense( b, x );
   }

   bool
   solveDense( ConstVectorViewType b, VectorViewType x )
   {
#ifdef __CUDACC__
      static_assert( Matrices::is_dense_matrix_v< Matrix >,
                     "CuSolverWrapper::solveDense can be called only for dense matrices." );

      TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

      const IndexType n = this->matrix->getRows();

      cusolverDnHandle_t handle;
      cusolverStatus_t status = cusolverDnCreate( &handle );
      if( status != CUSOLVER_STATUS_SUCCESS ) {
         std::cerr << "cusolverDnCreate failed: " << status << std::endl;
         cusolverDnDestroy( handle );
         return false;
      }

      x = b;
      Containers::Vector< int, Devices::Cuda > d_info( 1 );

      int lwork = 0;
      if constexpr( std::is_same_v< RealType, float > )
         status =
            cusolverDnSgetrf_bufferSize( handle, n, n, const_cast< float* >( this->matrix->getValues().getData() ), n, &lwork );
      else if constexpr( std::is_same_v< RealType, double > )
         status = cusolverDnDgetrf_bufferSize(
            handle, n, n, const_cast< double* >( this->matrix->getValues().getData() ), n, &lwork );

      Containers::Vector< RealType, Devices::Cuda > d_work( lwork );
      Containers::Vector< int, Devices::Cuda > d_pivots( n );

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
      return true;
#else
      throw std::runtime_error( "CuSolverWrapper was not built with CUDA support." );
#endif
   }

   bool
   solveSparse( ConstVectorViewType b, VectorViewType x )
   {
#ifdef __CUDACC__
      static_assert( Matrices::is_sparse_csr_matrix_v< Matrix >,
                     "CuSolverWrapper::solveSparse can be called only for sparse CSR matrices." );

      TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

      const IndexType size = this->matrix->getRows();

      this->resetIterations();
      this->setResidue( this->getConvergenceResidue() + 1.0 );

      cusolverSpHandle_t handle;
      cusolverStatus_t status;
      cusparseStatus_t cusparse_status;
      cusparseMatDescr_t mat_descr;

      status = cusolverSpCreate( &handle );
      if( status != CUSOLVER_STATUS_SUCCESS ) {
         std::cerr << "cusolverSpCreate failed: " << status << std::endl;
         cusparseDestroyMatDescr( mat_descr );
         cusolverSpDestroy( handle );
         return false;
      }

      cusparse_status = cusparseCreateMatDescr( &mat_descr );
      if( cusparse_status != CUSPARSE_STATUS_SUCCESS ) {
         std::cerr << "cusparseCreateMatDescr failed: " << cusparse_status << std::endl;
         cusparseDestroyMatDescr( mat_descr );
         cusolverSpDestroy( handle );
         return false;
      }

      cusparseSetMatType( mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL );
      cusparseSetMatIndexBase( mat_descr, CUSPARSE_INDEX_BASE_ZERO );

      const RealType tol = 1e-16;
      const int reorder = 0;
      int singularity = 0;

      if( std::is_same_v< typename Matrix::RealType, float > ) {
         status = cusolverSpScsrlsvqr( handle,
                                       size,
                                       this->matrix->getValues().getSize(),
                                       mat_descr,
                                       (const float*) this->matrix->getValues().getData(),
                                       this->matrix->getSegments().getOffsets().getData(),
                                       this->matrix->getColumnIndexes().getData(),
                                       (const float*) b.getData(),
                                       tol,
                                       reorder,
                                       (float*) x.getData(),
                                       &singularity );

         if( status != CUSOLVER_STATUS_SUCCESS ) {
            std::cerr << "cusolverSpScsrlsvqr failed: " << status << std::endl;
            cusparseDestroyMatDescr( mat_descr );
            cusolverSpDestroy( handle );
            return false;
         }
      }

      if( std::is_same_v< typename Matrix::RealType, double > ) {
         status = cusolverSpDcsrlsvqr( handle,
                                       size,
                                       this->matrix->getValues().getSize(),
                                       mat_descr,
                                       (const double*) this->matrix->getValues().getData(),
                                       this->matrix->getSegments().getOffsets().getData(),
                                       this->matrix->getColumnIndexes().getData(),
                                       (const double*) b.getData(),
                                       tol,
                                       reorder,
                                       (double*) x.getData(),
                                       &singularity );

         if( status != CUSOLVER_STATUS_SUCCESS ) {
            std::cerr << "cusolverSpDcsrlsvqr failed: " << status << std::endl;
            cusparseDestroyMatDescr( mat_descr );
            cusolverSpDestroy( handle );
            return false;
         }
      }

      cusparseDestroyMatDescr( mat_descr );
      cusolverSpDestroy( handle );
      return true;
#else
      throw std::runtime_error( "CuSolverWrapper was not built with CUDA support." );
#endif
   }
};

}  // namespace Solvers::Linear
}  // namespace TNL
