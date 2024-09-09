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

// Warp-level tile-based multiplication where each warp computes a portion of the result matrix tile
template< int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel4( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   // Define shared memory tiles
   __shared__ RealType tileA[ tileDim ][ tileDim ];
   __shared__ RealType tileB[ tileDim ][ tileDim ];

   // Calculate thread and block indices
   IndexType bx = blockIdx.x, by = blockIdx.y;
   IndexType tx = threadIdx.x, ty = threadIdx.y;

   // Calculate the row and column index
   IndexType row = by * tileDim + ty;
   IndexType col = bx * tileDim + tx;

   // Initialize the accumulator for C
   typename ResultMatrix::RealType CValue = 0;

   // Loop over the tiles of the input matrices
   for( IndexType m = 0; m < ( tileDim + matrixA.getColumns() - 1 ) / tileDim; ++m ) {
      // Load A and B tiles into shared memory
      if( m * tileDim + tx < matrixA.getColumns() && row < matrixA.getRows() )
         tileA[ ty ][ tx ] = matrixA( row, m * tileDim + tx );
      else
         tileA[ ty ][ tx ] = 0.0;

      if( m * tileDim + ty < matrixB.getRows() && col < matrixB.getColumns() )
         tileB[ ty ][ tx ] = matrixB( m * tileDim + ty, col );
      else
         tileB[ ty ][ tx ] = 0.0;

      __syncthreads();

      // Compute product for this tile
      for( IndexType k = 0; k < tileDim; ++k )
         CValue += tileA[ ty ][ k ] * tileB[ k ][ tx ];

      __syncthreads();
   }

   // Write the result to the global memory
   if( row < resultMatrix.getRows() && col < resultMatrix.getColumns() )
      resultMatrix( row, col ) = CValue * matrixMultiplicator;
#endif  //__CUDACC__
}

}  // namespace TNL::Benchmarks::DenseMatrices
