// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__
   #ifdef HAVE_CUTLASS

      #include <cutlass/cutlass.h>
      #include <cutlass/tensor_ref.h>
      #include <cutlass/gemm/device/gemm.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationCutlass( const DenseMatrix& matrix1, const DenseMatrix& matrix2, DenseMatrix& resultMatrix )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Cuda >, "This function is specialized for CUDA device only." );

   IndexType m = matrix1.getRows();
   IndexType n = matrix2.getColumns();
   IndexType k = matrix1.getColumns();

   using Layout = cutlass::layout::ColumnMajor;

   using CutlassGemm = cutlass::gemm::device::Gemm< RealType, Layout, RealType, Layout, RealType, Layout >;
   CutlassGemm gemm_operator;

   typename CutlassGemm::Arguments args( { m, n, k },                                // Problem size
                                         { matrix1.getValues().getData(), m },       // Leading dimension for A
                                         { matrix2.getValues().getData(), k },       // Leading dimension for B
                                         { resultMatrix.getValues().getData(), m },  // Leading dimension for C
                                         { resultMatrix.getValues().getData(), m },  // Leading dimension for C
                                         { 1.0, 0.0 }                                // alpha and beta
   );

   cutlass::Status status = gemm_operator( args );
   if( status != cutlass::Status::kSuccess ) {
      throw std::runtime_error( "CUTLASS GEMM failed" );
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices

   #endif  //HAVE_CUTLASS
#endif     //__CUDACC__
