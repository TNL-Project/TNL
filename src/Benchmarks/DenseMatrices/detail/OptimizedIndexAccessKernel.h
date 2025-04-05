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

// Optimized tile-based multiplication that improves memory access efficiency by calculating the linear thread index
template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel2( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator,
                       const typename ResultMatrix::IndexType gridIdx_x,
                       const typename ResultMatrix::IndexType gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Here we compute product C = A * B. To profit from the fast
   // shared memory we do it by tiles.
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   __shared__ RealType tileA[ tileDim * tileDim ];
   __shared__ RealType tileB[ tileDim * tileDim ];
   __shared__ RealType tileC[ tileDim * tileDim ];

   const IndexType& matrixARows = matrixA.getRows();
   const IndexType& matrixAColumns = matrixA.getColumns();
   const IndexType& matrixBRows = matrixB.getRows();
   const IndexType& matrixBColumns = matrixB.getColumns();
   IndexType row;
   IndexType col;
   // Reset the tile C
   for( row = 0; row < tileDim; row += tileRowBlockSize )
      tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] = 0.0;

   // Compute the result tile coordinates
   const IndexType resultTileRow = ( gridIdx_y * gridDim.y + blockIdx.y ) * tileDim;
   const IndexType resultTileColumn = ( gridIdx_x * gridDim.x + blockIdx.x ) * tileDim;

   // Sum over the matrix tiles
   for( IndexType i = 0; i < matrixAColumns; i += tileDim ) {
      IndexType linearThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
      row = linearThreadIdx / tileDim;
      col = linearThreadIdx % tileDim;

      const IndexType matrixARow = resultTileRow + row;
      const IndexType matrixAColumn = i + col;
      if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
         tileA[ row * tileDim + col ] = matrixA( matrixARow, matrixAColumn );

      const IndexType matrixBRow = i + row;
      const IndexType matrixBColumn = resultTileColumn + col;
      if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
         tileB[ row * tileDim + col ] = matrixB( matrixBRow, matrixBColumn );

      __syncthreads();

      //const IndexType tileALastRow = TNL::min( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = TNL::min( tileDim, matrixAColumns - i );

      for( IndexType j = 0; j < tileALastColumn; j++ )
         tileC[ row * tileDim + col ] += tileA[ row * tileDim + j ] * tileB[ j * tileDim + col ];

      __syncthreads();
   }

   // Write the result tile to the result matrix
   const IndexType& matrixCRows = resultMatrix.getRows();
   const IndexType& matrixCColumns = resultMatrix.getColumns();
   if( resultTileRow + row < matrixCRows && resultTileColumn + col < matrixCColumns )
      resultMatrix( resultTileRow + row, resultTileColumn + col ) = tileC[ row * tileDim + col ] * matrixMultiplicator;

#endif  //__CUDACC__
}

}  // namespace TNL::Benchmarks::DenseMatrices
