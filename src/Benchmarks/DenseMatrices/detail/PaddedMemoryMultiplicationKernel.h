// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Matrices/MatrixBase.h>

namespace TNL::Benchmarks::DenseMatrices {

// Warp-level tile-based multiplication kernel with padding in shared memory arrays to minimize bank conflicts
template< int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel5( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   // Define shared memory tiles with padding to avoid bank conflicts
   __shared__ RealType tileA[ tileDim ][ tileDim + 1 ];
   __shared__ RealType tileB[ tileDim ][ tileDim + 1 ];

   IndexType bx = blockIdx.x;
   IndexType by = blockIdx.y;
   IndexType tx = threadIdx.x;
   IndexType ty = threadIdx.y;

   IndexType row = by * tileDim + ty;
   IndexType col = bx * tileDim + tx;
   typename ResultMatrix::RealType CValue = 0;

   for( IndexType m = 0; m < ( tileDim + matrixA.getColumns() - 1 ) / tileDim; ++m ) {
      if( m * tileDim + tx < matrixA.getColumns() && row < matrixA.getRows() )
         tileA[ ty ][ tx ] = matrixA( row, m * tileDim + tx );
      else
         tileA[ ty ][ tx ] = 0.0;

      if( m * tileDim + ty < matrixB.getRows() && col < matrixB.getColumns() )
         tileB[ ty ][ tx ] = matrixB( m * tileDim + ty, col );
      else
         tileB[ ty ][ tx ] = 0.0;

      __syncthreads();

      // Unroll the loop for a fixed tile size
      #pragma unroll
      for( IndexType k = 0; k < tileDim; ++k ) {
         CValue += tileA[ ty ][ k ] * tileB[ k ][ tx ];
      }

      __syncthreads();
   }

   if( row < resultMatrix.getRows() && col < resultMatrix.getColumns() )
      resultMatrix( row, col ) = CValue * matrixMultiplicator;
#endif  //__CUDACC__
}

}  // namespace TNL::Benchmarks::DenseMatrices
