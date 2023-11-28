#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Matrices/MatrixBase.h>

namespace TNL::Benchmarks::DenseMatrices {

   //main kernel for dense matrix multiplication
   template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
   __global__
   void
   DenseMatrixProductKernel( ResultMatrix resultMatrix,
                           const Matrix1 matrixA,
                           const Matrix2 matrixB,
                           const typename ResultMatrix::RealType matrixMultiplicator,
                           const typename ResultMatrix::IndexType gridIdx_x,
                           const typename ResultMatrix::IndexType gridIdx_y )
   {
#ifdef __CUDACC__
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
               sum += matrixMultiplicator * tileA[ ( threadIdx.y + row ) * tileDim + j ] * tileB[ j * tileDim + threadIdx.x ];
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
            resultMatrix( matrixCRow, matrixCColumn ) = tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ];
      }
#endif
   }

   //kernel 2 (Optimizes the calculation of the linear thread index to access elements in the shared memory)
   template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
   __global__
   void OptimizedDenseMatrixProductKernel( ResultMatrix resultMatrix,
                                          const Matrix1 matrixA,
                                          const Matrix2 matrixB,
                                          const typename ResultMatrix::RealType matrixMultiplicator,
                                          const typename ResultMatrix::IndexType gridIdx_x,
                                          const typename ResultMatrix::IndexType gridIdx_y ) {
#ifdef __CUDACC__
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
      IndexType row, col;
      // Reset the tile C
      for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
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

         const IndexType tileALastRow = TNL::min( tileDim, matrixARows - resultTileRow );
         const IndexType tileALastColumn = TNL::min( tileDim, matrixAColumns - i );

         for( IndexType j = 0; j < tileALastColumn; j++ )
               tileC[ row * tileDim + col ] += matrixMultiplicator * tileA[ row * tileDim + j ] * tileB[ j * tileDim + col ];

         __syncthreads();
      }

      // Write the result tile to the result matrix
      const IndexType& matrixCRows = resultMatrix.getRows();
      const IndexType& matrixCColumns = resultMatrix.getColumns();
      if( resultTileRow + row < matrixCRows && resultTileColumn + col < matrixCColumns )
         resultMatrix( resultTileRow + row, resultTileColumn + col ) = tileC[ row * tileDim + col ];
#endif
   }

   //kernel 3 (Optimizes memory access patterns using 2D shared memory arrays instead of 1D arrays)
   template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
   __global__
   void Optimized2DenseMatrixProductKernel( ResultMatrix resultMatrix,
                                          const Matrix1 matrixA,
                                          const Matrix2 matrixB,
                                          const typename ResultMatrix::RealType matrixMultiplicator,
                                          const typename ResultMatrix::IndexType gridIdx_x,
                                          const typename ResultMatrix::IndexType gridIdx_y ) {
#ifdef __CUDACC__
      using IndexType = typename ResultMatrix::IndexType;
      using RealType = typename ResultMatrix::RealType;

      __shared__ RealType tileA[ tileDim ][ tileDim ];
      __shared__ RealType tileB[ tileDim ][ tileDim ];
      __shared__ RealType tileC[ tileDim ][ tileDim ];

      const IndexType matrixARows = matrixA.getRows();
      const IndexType matrixAColumns = matrixA.getColumns();
      const IndexType matrixBRows = matrixB.getRows();
      const IndexType matrixBColumns = matrixB.getColumns();

      IndexType row, col;

      // Reset the tile C
      for (IndexType r = 0; r < tileDim; r += tileRowBlockSize)
         tileC[r + threadIdx.y][threadIdx.x] = 0.0;

      // Compute the result tile coordinates
      const IndexType resultTileRow = (gridIdx_y * gridDim.y + blockIdx.y) * tileDim;
      const IndexType resultTileColumn = (gridIdx_x * gridDim.x + blockIdx.x) * tileDim;

      // Sum over the matrix tiles
      for (IndexType i = 0; i < matrixAColumns; i += tileDim) {
         row = threadIdx.y;
         col = threadIdx.x;

         const IndexType matrixARow = resultTileRow + row;
         const IndexType matrixAColumn = i + col;

         if (matrixARow < matrixARows && matrixAColumn < matrixAColumns)
               tileA[row][col] = matrixA(matrixARow, matrixAColumn);

         const IndexType matrixBRow = i + row;
         const IndexType matrixBColumn = resultTileColumn + col;

         if (matrixBRow < matrixBRows && matrixBColumn < matrixBColumns)
               tileB[row][col] = matrixB(matrixBRow, matrixBColumn);

         __syncthreads();

         const IndexType tileALastRow = min(tileDim, matrixARows - resultTileRow);
         const IndexType tileALastColumn = min(tileDim, matrixAColumns - i);

         for (IndexType j = 0; j < tileALastColumn; j++)
               tileC[row][col] += matrixMultiplicator * tileA[row][j] * tileB[j][col];

         __syncthreads();
      }

      // Write the result tile to the result matrix
      const IndexType matrixCRows = resultMatrix.getRows();
      const IndexType matrixCColumns = resultMatrix.getColumns();

      if (resultTileRow + threadIdx.y < matrixCRows && resultTileColumn + threadIdx.x < matrixCColumns)
         resultMatrix(resultTileRow + threadIdx.y, resultTileColumn + threadIdx.x) = tileC[threadIdx.y][threadIdx.x];
#endif
   }

   //kernel 4 (each warp is responsible for computing a subset of a tile)
   template< int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2 >
   __global__ void WarpTilingDenseMatrixProductKernel(ResultMatrix resultMatrix,
                                                      const Matrix1 matrixA,
                                                      const Matrix2 matrixB,
                                                      const typename ResultMatrix::RealType matrixMultiplicator) {
#ifdef __CUDACC__
      // Define shared memory tiles
      __shared__ typename ResultMatrix::RealType tileA[tileDim][tileDim];
      __shared__ typename ResultMatrix::RealType tileB[tileDim][tileDim];

      // Calculate thread and block indices
      int bx = blockIdx.x, by = blockIdx.y;
      int tx = threadIdx.x, ty = threadIdx.y;

      // Calculate the row and column index
      int row = by * tileDim + ty;
      int col = bx * tileDim + tx;

      // Initialize the accumulator for C
      typename ResultMatrix::RealType CValue = 0;

      // Loop over the tiles of the input matrices
      for (int m = 0; m < (tileDim + matrixA.getColumns() - 1)/tileDim; ++m) {
         // Load A and B tiles into shared memory
         if (m * tileDim + tx < matrixA.getColumns() && row < matrixA.getRows())
               tileA[ty][tx] = matrixA(row, m * tileDim + tx);
         else
               tileA[ty][tx] = 0.0;

         if (m * tileDim + ty < matrixB.getRows() && col < matrixB.getColumns())
               tileB[ty][tx] = matrixB(m * tileDim + ty, col);
         else
               tileB[ty][tx] = 0.0;

         __syncthreads();

         // Compute product for this tile
         for (int k = 0; k < tileDim; ++k)
               CValue += tileA[ty][k] * tileB[k][tx];

         __syncthreads();
      }

      // Write the result to the global memory
      if (row < resultMatrix.getRows() && col < resultMatrix.getColumns())
         resultMatrix(row, col) = CValue * matrixMultiplicator;
#endif
   }

   //kernel 5 (padding in shared memory to reduce conflicts in shared memory)
   template< int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2 >
   __global__ void OptimizedWarpTilingDenseMatrixProductKernel(ResultMatrix resultMatrix,
                                                               const Matrix1 matrixA,
                                                               const Matrix2 matrixB,
                                                               const typename ResultMatrix::RealType matrixMultiplicator) {
#ifdef __CUDACC__
      // Define shared memory tiles with padding to avoid bank conflicts
      __shared__ typename ResultMatrix::RealType tileA[tileDim][tileDim + 1];
      __shared__ typename ResultMatrix::RealType tileB[tileDim][tileDim + 1];

      int bx = blockIdx.x, by = blockIdx.y;
      int tx = threadIdx.x, ty = threadIdx.y;

      int row = by * tileDim + ty;
      int col = bx * tileDim + tx;
      typename ResultMatrix::RealType CValue = 0;

      for (int m = 0; m < (tileDim + matrixA.getColumns() - 1) / tileDim; ++m) {
         if (m * tileDim + tx < matrixA.getColumns() && row < matrixA.getRows())
               tileA[ty][tx] = matrixA(row, m * tileDim + tx);
         else
               tileA[ty][tx] = 0.0;

         if (m * tileDim + ty < matrixB.getRows() && col < matrixB.getColumns())
               tileB[ty][tx] = matrixB(m * tileDim + ty, col);
         else
               tileB[ty][tx] = 0.0;

         __syncthreads();

         // Unroll the loop for a fixed tile size
   #pragma unroll
         for (int k = 0; k < tileDim; ++k) {
               CValue += tileA[ty][k] * tileB[k][tx];
         }

         __syncthreads();
      }

      if (row < resultMatrix.getRows() && col < resultMatrix.getColumns())
         resultMatrix(row, col) = CValue * matrixMultiplicator;
#endif
   }
}


