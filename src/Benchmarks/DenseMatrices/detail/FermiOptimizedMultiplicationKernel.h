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

// Fermi-architecture optimized multiplication without shared memory, computing 4x4 blocks for efficient performance
template< typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel6( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator )
{
#ifdef __CUDACC__
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   // Shared memory for submatrices of A and B
   __shared__ RealType sharedA[ 64 ][ 17 ];  // Padding to prevent bank conflicts
   __shared__ RealType sharedB[ 16 ][ 65 ];  // Padding to prevent bank conflicts

   // Each thread computes a 4x4 block of the result matrix
   IndexType row = blockIdx.y * 64 + threadIdx.y * 4;
   IndexType col = blockIdx.x * 64 + threadIdx.x * 4;

   RealType CValue[ 4 ][ 4 ] = { { 0 } };

   // Bounds
   const IndexType matrixARows = matrixA.getRows();
   const IndexType matrixAColumns = matrixA.getColumns();
   const IndexType matrixBRows = matrixB.getRows();
   const IndexType matrixBColumns = matrixB.getColumns();

   const auto& AValues = matrixA.getValues();
   const auto& BValues = matrixB.getValues();
   auto& resultValues = resultMatrix.getValues();

   // Compute number of tiles
   const IndexType numTiles = ( matrixAColumns + 15 ) / 16;

   // Precompute the maximum valid iterations for aRow and bCol
   IndexType maxARow = min( 4, matrixARows - row );
   IndexType maxBCol = min( 4, matrixBColumns - col );

   // Iterate through each tile
   for( IndexType m = 0; m < numTiles; ++m ) {
      IndexType aCol = m * 16 + threadIdx.x;
      IndexType bRow = m * 16 + threadIdx.y;

      // Load valid data from global memory into shared memory for matrix A
      if( aCol < matrixAColumns ) {
         for( IndexType i = 0; i < maxARow; ++i ) {
            IndexType aRow = row + i;
            sharedA[ threadIdx.y * 4 + i ][ threadIdx.x ] = AValues[ aRow + aCol * matrixARows ];
         }
         for( IndexType i = maxARow; i < 4; ++i ) {
            sharedA[ threadIdx.y * 4 + i ][ threadIdx.x ] = 0.0;
         }
      }
      else {
         #pragma unroll
         for( IndexType i = 0; i < 4; ++i ) {
            sharedA[ threadIdx.y * 4 + i ][ threadIdx.x ] = 0.0;
         }
      }

      // Load valid data from global memory into shared memory for matrix B
      if( bRow < matrixBRows ) {
         for( IndexType i = 0; i < maxBCol; ++i ) {
            IndexType bCol = col + i;
            sharedB[ threadIdx.y ][ threadIdx.x * 4 + i ] = BValues[ bRow + bCol * matrixBRows ];
         }
         for( IndexType i = maxBCol; i < 4; ++i ) {
            sharedB[ threadIdx.y ][ threadIdx.x * 4 + i ] = 0.0;
         }
      }
      else {
         #pragma unroll
         for( IndexType i = 0; i < 4; ++i ) {
            sharedB[ threadIdx.y ][ threadIdx.x * 4 + i ] = 0.0;
         }
      }
      __syncthreads();

      // Compute the matrix multiplication for this tile
      for( IndexType k = 0; k < 16; ++k ) {
         RealType regA[ 4 ];
         RealType regB[ 4 ];
         #pragma unroll
         for( IndexType i = 0; i < 4; ++i ) {
            regA[ i ] = sharedA[ threadIdx.y * 4 + i ][ k ];
            regB[ i ] = sharedB[ k ][ threadIdx.x * 4 + i ];
         }
         for( IndexType i = 0; i < 4; ++i ) {
            #pragma unroll
            for( IndexType j = 0; j < 4; ++j ) {
               CValue[ i ][ j ] += regA[ i ] * regB[ j ];
            }
         }
      }
      __syncthreads();
   }

   // Store the result into the result matrix
   for( IndexType i = 0; i < maxARow; ++i ) {
      for( IndexType j = 0; j < maxBCol; ++j ) {
         IndexType cRow = row + i;
         IndexType cCol = col + j;
         IndexType index = cRow + cCol * matrixARows;
         resultValues[ index ] = CValue[ i ][ j ] * matrixMultiplicator;
      }
   }
#endif  // __CUDACC__
}

}  // namespace TNL::Benchmarks::DenseMatrices
