// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __CUDACC__

   #include <cusolverSp.h>
   #include <cusolverDn.h>

   #include <TNL/Config/ConfigDescription.h>
   #include <TNL/Matrices/SparseMatrix.h>
   #include <TNL/Matrices/DenseMatrix.h>
   #include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {

template< typename Matrix >
struct is_csr_matrix_type
{
   static constexpr bool value = false;
};

template< typename Real, typename Device, typename Index, typename ComputeReal, typename RealAllocator, typename IndexAllocator >
struct is_csr_matrix_type< TNL::Matrices::SparseMatrix< Real,
                                                        Device,
                                                        Index,
                                                        Matrices::GeneralMatrix,
                                                        Algorithms::Segments::CSR,
                                                        ComputeReal,
                                                        RealAllocator,
                                                        IndexAllocator > >
{
   static constexpr bool value = true;
};

template< typename Matrix >
constexpr bool is_csr_matrix_type_v = is_csr_matrix_type< Matrix >::value;

namespace Solvers::Linear {

template< typename Matrix >
class CuSolverWrapper : public LinearSolver< Matrix >
{
   static_assert( is_csr_matrix_type_v< Matrix > || Matrices::is_dense_matrix_type_v< Matrix >,
                  "The CuSolverWrapper solver is available only for CSR and dense matrices." );
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value,
                  "CuSolverWrapper is available only on CUDA" );
   static_assert( std::is_same< typename Matrix::RealType, float >::value
                     || std::is_same< typename Matrix::RealType, double >::value,
                  "unsupported RealType" );
   static_assert( std::is_same< typename Matrix::IndexType, int >::value, "unsupported IndexType" );

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
      if constexpr( is_csr_matrix_type_v< Matrix > )
         return solveSparse( b, x );
      else  // is_dense_matrix_v< Matrix > must be true
         return solveDense( b, x );
   }

   bool
   solveDense( ConstVectorViewType b, VectorViewType x )
   {
      TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
      TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

      const IndexType n = this->matrix->getRows();

      cusolverDnHandle_t handle;
      cusolverStatus_t status = cusolverDnCreate( &handle );
      if( status != CUSOLVER_STATUS_SUCCESS ) {
         std::cerr << "cusolverDnCreate failed: " << status << std::endl;
         return false;
      }

      x = b;
      int* d_info;
      cudaMalloc( &d_info, sizeof( int ) );

      int lwork = 0;
      if constexpr( std::is_same< RealType, float >::value )
         status =
            cusolverDnSgetrf_bufferSize( handle, n, n, const_cast< float* >( this->matrix->getValues().getData() ), n, &lwork );
      else if constexpr( std::is_same< RealType, double >::value )
         status = cusolverDnDgetrf_bufferSize(
            handle, n, n, const_cast< double* >( this->matrix->getValues().getData() ), n, &lwork );

      Containers::Vector< RealType, Devices::Cuda > d_work( lwork );
      Containers::Vector< int, Devices::Cuda > d_pivots( n );

      // LU factorization
      if constexpr( std::is_same< RealType, float >::value )
         status = cusolverDnSgetrf( handle,
                                    n,
                                    n,
                                    const_cast< float* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_work.getData(),
                                    d_pivots.getData(),
                                    d_info );
      else if constexpr( std::is_same< RealType, double >::value )
         status = cusolverDnDgetrf( handle,
                                    n,
                                    n,
                                    const_cast< double* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_work.getData(),
                                    d_pivots.getData(),
                                    d_info );

      // Solve Ax = b
      if constexpr( std::is_same< RealType, float >::value )
         status = cusolverDnSgetrs( handle,
                                    CUBLAS_OP_N,
                                    n,
                                    1,
                                    const_cast< float* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_pivots.getData(),
                                    x.getData(),
                                    n,
                                    d_info );
      else if constexpr( std::is_same< RealType, double >::value )
         status = cusolverDnDgetrs( handle,
                                    CUBLAS_OP_N,
                                    n,
                                    1,
                                    const_cast< double* >( this->matrix->getValues().getData() ),
                                    n,
                                    d_pivots.getData(),
                                    x.getData(),
                                    n,
                                    d_info );

      cudaFree( d_info );
      return true;
   }

   bool
   solveSparse( ConstVectorViewType b, VectorViewType x )
   {
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
         return false;
      }

      cusparse_status = cusparseCreateMatDescr( &mat_descr );
      if( cusparse_status != CUSPARSE_STATUS_SUCCESS ) {
         std::cerr << "cusparseCreateMatDescr failed: " << cusparse_status << std::endl;
         return false;
      }

      cusparseSetMatType( mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL );
      cusparseSetMatIndexBase( mat_descr, CUSPARSE_INDEX_BASE_ZERO );

      const RealType tol = 1e-16;
      const int reorder = 0;
      int singularity = 0;

      if( std::is_same< typename Matrix::RealType, float >::value ) {
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
            return false;
         }
      }

      if( std::is_same< typename Matrix::RealType, double >::value ) {
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
            return false;
         }
      }

      cusparseDestroyMatDescr( mat_descr );
      cusolverSpDestroy( handle );

      return true;
   }
};

}  // namespace Solvers::Linear
}  // namespace TNL

#endif
