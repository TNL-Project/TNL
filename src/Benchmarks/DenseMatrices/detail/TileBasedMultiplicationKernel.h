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

// Previous main kernel for dense matrix multiplication
// Utilizing tile-based dense matrix multiplication using shared memory for improved performance
template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel1( ResultMatrix resultMatrix,
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

   // Reset the tile C
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
      tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] = 0.0;

   // Compute the result tile coordinates
   const IndexType resultTileRow = ( gridIdx_y * gridDim.y + blockIdx.y ) * tileDim;
   const IndexType resultTileColumn = ( gridIdx_x * gridDim.x + blockIdx.x ) * tileDim;

   // Sum over the matrix tiles
   for( IndexType i = 0; i < matrixAColumns; i += tileDim ) {
      for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
         const IndexType matrixARow = resultTileRow + threadIdx.y + row;
         const IndexType matrixAColumn = i + threadIdx.x;
         if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
            tileA[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] = matrixA( matrixARow, matrixAColumn );

         const IndexType matrixBRow = i + threadIdx.y + row;
         const IndexType matrixBColumn = resultTileColumn + threadIdx.x;
         if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
            tileB[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] = matrixB( matrixBRow, matrixBColumn );
      }
      __syncthreads();

      const IndexType tileALastRow = TNL::min( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = TNL::min( tileDim, matrixAColumns - i );
      // const IndexType tileBLastRow = TNL::min( tileDim, matrixBRows - i );
      // const IndexType tileBLastColumn = TNL::min( tileDim, matrixBColumns - resultTileColumn );

      for( IndexType row = 0; row < tileALastRow; row += tileRowBlockSize ) {
         RealType sum( 0.0 );
         for( IndexType j = 0; j < tileALastColumn; j++ )
            sum += tileA[ ( threadIdx.y + row ) * tileDim + j ] * tileB[ j * tileDim + threadIdx.x ];
         tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] += sum;
      }
      __syncthreads();
   }

   // Write the result tile to the result matrix
   const IndexType& matrixCRows = resultMatrix.getRows();
   const IndexType& matrixCColumns = resultMatrix.getColumns();
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
      const IndexType matrixCRow = resultTileRow + row + threadIdx.y;
      const IndexType matrixCColumn = resultTileColumn + threadIdx.x;
      if( matrixCRow < matrixCRows && matrixCColumn < matrixCColumns )
         resultMatrix( matrixCRow, matrixCColumn ) =
            tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] * matrixMultiplicator;
   }
#endif  //__CUDACC__
}

}  // namespace TNL::Benchmarks::DenseMatrices
