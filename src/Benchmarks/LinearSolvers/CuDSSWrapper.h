// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_CUDSS
   #include <cudss.h>
#endif

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL::Solvers::Linear {

template< typename Matrix, typename SolverMonitor = DirectSolverMonitor< double, int > >
class CuDSSWrapper : public DirectSolver< typename Matrix::RealType, typename Matrix::IndexType, SolverMonitor >
{
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda >, "CuDSSWrapper is available only on CUDA" );
   static_assert( std::is_same_v< typename Matrix::RealType, float > || std::is_same_v< typename Matrix::RealType, double >,
                  "unsupported RealType" );
   static_assert( std::is_same_v< typename Matrix::IndexType, int >, "unsupported IndexType" );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using MatrixPointer = typename Base::MatrixPointer;

   void
   setMatrix( const MatrixPointer& matrix )
   {
#ifdef HAVE_CUDSS
      this->size = matrix->getRows();
      cudssCreate( &handle );
      TNL_CHECK_CUDA_DEVICE;
      cudssConfigCreate( &solverConfig );
      TNL_CHECK_CUDA_DEVICE;
      cudssDataCreate( handle, &solverData );
      TNL_CHECK_CUDA_DEVICE;
      if( std::is_same< RealType, float >::value )
         valueType = CUDA_R_32F;
      if( std::is_same< RealType, double >::value )
         valueType = CUDA_R_64F;
      cudssMatrixCreateCsr( &A,
                            matrix->getRows(),
                            matrix->getColumns(),
                            matrix->getNonzeroElementsCount(),
                            (void*) matrix->getSegments().getOffsets().getData(),
                            nullptr,
                            (void*) matrix->getColumnIndexes().getData(),
                            (void*) matrix->getValues().getData(),
                            CUDA_R_32I,
                            valueType,
                            CUDSS_MTYPE_GENERAL,
                            CUDSS_MVIEW_FULL,
                            CUDSS_BASE_ZERO );

      cudssExecute( handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b );
      TNL_CHECK_CUDA_DEVICE;

      cudssExecute( handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b );
      TNL_CHECK_CUDA_DEVICE;

      this->factorisation_success = true;
#endif
   }

   bool
   solve( ConstVectorViewType b_, VectorViewType x_ )
   {
#ifdef HAVE_CUDSS
      cudssMatrixCreateDn( &b, this->size, 1, size, (void*) b_.getData(), valueType, CUDSS_LAYOUT_COL_MAJOR );
      TNL_CHECK_CUDA_DEVICE;
      cudssMatrixCreateDn( &x, this->size, 1, size, (void*) x_.getData(), valueType, CUDSS_LAYOUT_COL_MAJOR );
      TNL_CHECK_CUDA_DEVICE;

      cudssExecute( handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b );
      TNL_CHECK_CUDA_DEVICE;
      this->solver_success = true;
      return true;
#else
      std::cerr << "CuDSS is not available." << std::endl;
      return false;
#endif
   }

   bool
   succeeded() const
   {
      return this->solver_success;
   }

   bool
   factorized() const
   {
      return this->factorisation_success;
   }

protected:
#ifdef HAVE_CUDSS
   cudssHandle_t handle;
   cudaDataType_t valueType;
   cudssConfig_t solverConfig;
   cudssData_t solverData;
   cudssMatrix_t A;
   cudssMatrix_t x, b;
   IndexType size;
   bool factorisation_success = false;
   bool solver_success = false;
#endif
};

}  // namespace TNL::Solvers::Linear
