#pragma once

#ifdef __CUDACC__
   #include "GEMkernels.h"

namespace TNL::Solvers::Linear::detail {
template< typename Matrix, typename Vector >
void
calculResultVector( const Matrix& matrix, const Vector& device_vector, Vector& x )
{
   //Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( matrix );

   int blockSize = matrix.getRows() > 256 ? 256 : matrix.getColumns();
   int numBlocksOnColumn = TNL::roundUpDivision( matrix.getRows(), 256 );
   int numOfBlocks = matrix.getRows() * numBlocksOnColumn;

   // clang-format off
   GEMDiagToResult<<< numOfBlocks, blockSize >>>( matrix.getConstView(), device_vector.getConstView(), x.getView() );
   // clang-format on
   cudaDeviceSynchronize();
   TNL_CHECK_CUDA_DEVICE;
}
}  // namespace TNL::Solvers::Linear::detail

#endif
