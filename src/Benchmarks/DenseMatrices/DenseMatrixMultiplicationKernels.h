#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Matrices/MatrixBase.h>

#ifdef __CUDACC__
   #ifdef USE_TENSOR_CORES
      #include <mma.h>
   #endif
#endif

namespace TNL::Benchmarks::DenseMatrices {

//previous main kernel for dense matrix multiplication
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
#endif  //__CUDACC__
}

//kernel 2 (Optimizes the calculation of the linear thread index to access elements in the shared memory)
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
   IndexType row, col;
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
#endif  //__CUDACC__
}

//kernel 3 (Optimizes memory access patterns using 2D shared memory arrays instead of 1D arrays)
template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel3( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator,
                       const typename ResultMatrix::IndexType gridIdx_x,
                       const typename ResultMatrix::IndexType gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
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
   for( IndexType r = 0; r < tileDim; r += tileRowBlockSize )
      tileC[ r + threadIdx.y ][ threadIdx.x ] = 0.0;

   // Compute the result tile coordinates
   const IndexType resultTileRow = ( gridIdx_y * gridDim.y + blockIdx.y ) * tileDim;
   const IndexType resultTileColumn = ( gridIdx_x * gridDim.x + blockIdx.x ) * tileDim;

   // Sum over the matrix tiles
   for( IndexType i = 0; i < matrixAColumns; i += tileDim ) {
      row = threadIdx.y;
      col = threadIdx.x;

      const IndexType matrixARow = resultTileRow + row;
      const IndexType matrixAColumn = i + col;

      if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
         tileA[ row ][ col ] = matrixA( matrixARow, matrixAColumn );

      const IndexType matrixBRow = i + row;
      const IndexType matrixBColumn = resultTileColumn + col;

      if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
         tileB[ row ][ col ] = matrixB( matrixBRow, matrixBColumn );

      __syncthreads();

      const IndexType tileALastRow = min( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = min( tileDim, matrixAColumns - i );

      for( IndexType j = 0; j < tileALastColumn; j++ )
         tileC[ row ][ col ] += matrixMultiplicator * tileA[ row ][ j ] * tileB[ j ][ col ];

      __syncthreads();
   }

   // Write the result tile to the result matrix
   const IndexType matrixCRows = resultMatrix.getRows();
   const IndexType matrixCColumns = resultMatrix.getColumns();

   if( resultTileRow + threadIdx.y < matrixCRows && resultTileColumn + threadIdx.x < matrixCColumns )
      resultMatrix( resultTileRow + threadIdx.y, resultTileColumn + threadIdx.x ) = tileC[ threadIdx.y ][ threadIdx.x ];
#endif  //__CUDACC__
}

//kernel 4 (each warp is responsible for computing a subset of a tile)
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

//kernel 5 (padding in shared memory to reduce conflicts)
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

   IndexType bx = blockIdx.x, by = blockIdx.y;
   IndexType tx = threadIdx.x, ty = threadIdx.y;

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

/*
//kernel 6 (Fermi)
template< typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
optimizedFermiGemmKernel( ResultMatrix resultMatrix,
                          const Matrix1 matrixA,
                          const Matrix2 matrixB,
                          const typename ResultMatrix::RealType matrixMultiplicator )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   // Define shared memory tiles
   __shared__ RealType tileA[ 64 ][ 16 + 1 ];
   __shared__ RealType tileB[ 16 ][ 64 + 1 ];

   auto matrixARows = matrixA.getRows();
   auto matrixAColumns = matrixA.getColumns();
   auto matrixBRows = matrixB.getRows();
   auto matrixBColumns = matrixB.getColumns();
   auto matrixCRows = resultMatrix.getRows();
   auto matrixCColumns = resultMatrix.getColumns();

   IndexType bx = blockIdx.x, by = blockIdx.y;
   IndexType tx = threadIdx.x, ty = threadIdx.y;

   // Each thread computes a 4x4 sub-matrix
   IndexType row = by * 64 + ty * 4;
   IndexType col = bx * 64 + tx * 4;

   RealType CValue[ 4 ][ 4 ] = { 0 };

   for( IndexType m = 0; m < ( matrixA.getColumns() + 15 ) / 16; ++m ) {
      // Load tiles from A and B into shared memory
      for( IndexType i = 0; i < 4; i++ ) {
         IndexType aRow = row + i;
         IndexType bCol = col + i;
         IndexType aCol = m * 16 + tx;
         IndexType bRow = m * 16 + ty;

         // Check and load for tileA
         if( aRow < matrixARows && aCol < matrixAColumns ) {
            tileA[ ty * 4 + i ][ tx ] = matrixA( aRow, aCol );
         }
         else {
            tileA[ ty * 4 + i ][ tx ] = 0.0;
         }

         // Check and load for tileB
         if( bRow < matrixBRows && bCol < matrixBColumns ) {
            tileB[ ty ][ tx * 4 + i ] = matrixB( bRow, bCol );
         }
         else {
            tileB[ ty ][ tx * 4 + i ] = 0.0;
         }
      }

      __syncthreads();

      // Perform the multiplication for this tile
      for( IndexType k = 0; k < 16; ++k ) {
         RealType regA[ 4 ], regB[ 4 ];  //Assing registers computing 8 elements
         // Load elements from shared memory into registers
         for( IndexType i = 0; i < 4; ++i ) {
            regA[ i ] = tileA[ ty * 4 + i ][ k ];
            regB[ i ] = tileB[ k ][ tx * 4 + i ];
         }
         // Compute using the loaded elements
         for( IndexType i = 0; i < 4; ++i ) {
            for( IndexType j = 0; j < 4; ++j ) {
               CValue[ i ][ j ] += regA[ i ] * regB[ j ] * matrixMultiplicator;
            }
         }
      }
      __syncthreads();
   }

   // Write the result to the global memory
   for( IndexType i = 0; i < 4; i++ ) {
      for( IndexType j = 0; j < 4; j++ ) {
         IndexType cRow = row + i;
         IndexType cCol = col + j;
         if( cRow < matrixCRows && cCol < matrixCColumns ) {
            resultMatrix( cRow, cCol ) = CValue[ i ][ j ] * matrixMultiplicator;
         }
      }
   }
#endif  // __CUDACC__
}
*/

//kernel 6 (Fermi)
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

   IndexType bx = blockIdx.x, by = blockIdx.y;
   IndexType tx = threadIdx.x, ty = threadIdx.y;

   IndexType row = by * 64 + ty * 4;  // Each thread computes 4 rows at a time
   IndexType col = bx * 64 + tx * 4;  // Each thread computes 4 columns at a time

   RealType CValue[ 4 ][ 4 ] = { 0 };  // Accumulator for the result sub-tile

   const IndexType matrixARows = matrixA.getRows();
   const IndexType matrixAColumns = matrixA.getColumns();
   const IndexType matrixBRows = matrixB.getRows();
   const IndexType matrixBColumns = matrixB.getColumns();

   const auto& AValues = matrixA.getValues();      // Vector of A's elements
   const auto& BValues = matrixB.getValues();      // Vector of B's elements
   auto& resultValues = resultMatrix.getValues();  // Vector of ResultMatrix's elements

   // Calculate number of phases required to cover all columns of A / rows of B
   const IndexType numTiles = ( matrixAColumns + 1 ) / 2;

   // Precompute row and column boundary checks
   const IndexType maxRowIndexA = min( matrixARows - 1, row + 3 );
   const IndexType maxColIndexB = min( matrixBColumns - 1, col + 3 );

   for( IndexType m = 0; m < numTiles; ++m ) {
      RealType AReg[ 4 ][ 2 ] = { { 0 } };
      RealType BReg[ 4 ][ 2 ] = { { 0 } };

      IndexType maxKForA = min( 1, matrixAColumns - 1 - m * 2 );
      IndexType maxKForB = min( 1, matrixBRows - 1 - m * 2 );

      // Load data into registers
      for( IndexType k = 0; k <= maxKForA; ++k ) {
   #pragma unroll
         for( IndexType i = 0; i <= maxRowIndexA - row; ++i ) {
            AReg[ i ][ k ] = AValues[ ( m * 2 + k ) * matrixARows + ( row + i ) ];
         }
      }

      for( IndexType k = 0; k <= maxKForB; ++k ) {
   #pragma unroll
         for( IndexType j = 0; j <= maxColIndexB - col; ++j ) {
            BReg[ j ][ k ] = BValues[ ( col + j ) * matrixBRows + ( m * 2 + k ) ];
         }
      }

      // Matrix multiplication for the current tile
      for( IndexType k = 0; k < 2; ++k ) {
         for( IndexType i = 0; i < 4; ++i ) {
   #pragma unroll
            for( IndexType j = 0; j < 4; ++j ) {
               CValue[ i ][ j ] += AReg[ i ][ k ] * BReg[ j ][ k ] * matrixMultiplicator;
            }
         }
      }
   }

   // Store the result from CValue to the result matrix
   for( IndexType i = 0; i < 4; i++ ) {
   #pragma unroll
      for( IndexType j = 0; j < 4; j++ ) {
         IndexType cRow = row + i;
         IndexType cCol = col + j;
         if( cRow < resultMatrix.getRows() && cCol < resultMatrix.getColumns() ) {
            IndexType index = cCol * resultMatrix.getRows() + cRow;
            resultValues[ index ] = CValue[ i ][ j ];
         }
      }
   }
#endif
}

//kernel 7 (Tensor Cores Utilization)
template< typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
MultiplicationKernel7( ResultMatrix resultMatrix,
                       const Matrix1 matrixA,
                       const Matrix2 matrixB,
                       const typename ResultMatrix::RealType matrixMultiplicator )
{
#ifdef __CUDACC__
   #ifdef USE_TENSOR_CORES
   using namespace nvcuda;
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   const IndexType WMMA_M = 16;
   const IndexType WMMA_N = 16;
   const IndexType WMMA_K = 16;

   const IndexType M = matrixA.getRows();
   const IndexType N = matrixB.getColumns();
   const IndexType K = matrixB.getRows();

   // Calculate the indices for the warps
   IndexType warpM = ( blockIdx.x * blockDim.x + threadIdx.x ) / warpSize;
   IndexType warpN = ( blockIdx.y * blockDim.y + threadIdx.y );

   // Declare the fragments
   wmma::fragment< wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major > a_frag;
   wmma::fragment< wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major > b_frag;
   wmma::fragment< wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float > acc_frag;
   wmma::fragment< wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float > c_frag;

   // Initialize the accumulator
   wmma::fill_fragment( acc_frag, 0.0f );

   // Compute each tile
   for( IndexType tile_k = 0; tile_k < K; tile_k += WMMA_K ) {
      IndexType aRow = warpM * WMMA_M;
      IndexType bCol = warpN * WMMA_N;

      if( aRow < M && bCol < N ) {
         for( IndexType step_k = 0; step_k < WMMA_K; step_k++ ) {
            if( ( tile_k + step_k ) < K ) {
               const half* aPtr = reinterpret_cast< const half* >( matrixA.getValues().getData() + aRow * K + tile_k + step_k );
               const half* bPtr =
                  reinterpret_cast< const half* >( matrixB.getValues().getData() + ( tile_k + step_k ) * N + bCol );

               // Load the matrix tiles into the fragments
               wmma::load_matrix_sync( a_frag, aPtr, K );
               wmma::load_matrix_sync( b_frag, bPtr, N );

               // Perform the matrix multiplication
               wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );
            }
         }
      }

      // Load in current value of C and combine it with the result of multiplication
      IndexType cRow = warpM * WMMA_M;
      IndexType cCol = warpN * WMMA_N;

      if( cRow < M && cCol < N ) {
         float* cPtr = reinterpret_cast< float* >( resultMatrix.getValues().getData() + cRow * N + cCol );
         wmma::load_matrix_sync( c_frag, cPtr, N, wmma::mem_col_major );

         for( IndexType i = 0; i < c_frag.num_elements; ++i ) {
            c_frag.x[ i ] = acc_frag.x[ i ] * matrixMultiplicator + c_frag.x[ i ];
         }

         // Store the updated value back to the result matrix
         wmma::store_matrix_sync( cPtr, c_frag, N, wmma::mem_col_major );
      }
   }
   #endif
#endif
}

}  //namespace TNL::Benchmarks::DenseMatrices
