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

template< int tileDim, int tileRowBlockSize, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionAlignedKernel( OutputMatrix resultMatrix,
                                 const InputMatrix inputMatrix,
                                 const Real matrixMultiplicator,
                                 const Index gridIdx_x,
                                 const Index gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix.getColumns();
   const Index rows = inputMatrix.getRows();

   // Diagonal mapping of the CUDA blocks
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   // Read the tile to the shared memory
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      tile[ Backend::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
         inputMatrix( readRowPosition + rowBlock, readColumnPosition );
   }
   __syncthreads();

   // Write the tile to the global memory
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      resultMatrix( writeRowPosition + rowBlock, writeColumnPosition ) =
         matrixMultiplicator * tile[ Backend::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ];
   }
#endif
}

template< int tileDim, int tileRowBlockSize, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionNonAlignedKernel( OutputMatrix resultMatrix,
                                    const InputMatrix inputMatrix,
                                    const Real matrixMultiplicator,
                                    const Index gridIdx_x,
                                    const Index gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix.getColumns();
   const Index rows = inputMatrix.getRows();

   // Diagonal mapping of the CUDA blocks
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   // Read the tile to the shared memory
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   if( readColumnPosition < columns ) {
      // const Index readOffset = readRowPosition * columns + readColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( readRowPosition + rowBlock < rows )
            tile[ Backend::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix( readRowPosition + rowBlock, readColumnPosition );
      }
   }
   __syncthreads();

   // Write the tile to the global memory
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   if( writeColumnPosition < rows ) {
      // const Index writeOffset = writeRowPosition * rows + writeColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( writeRowPosition + rowBlock < columns )
            resultMatrix( writeRowPosition + rowBlock, writeColumnPosition ) =
               matrixMultiplicator * tile[ Backend::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ];
      }
   }
#endif
}

// Combined kernel for both aligned and not-aligned transpositon kernels
// In the launching we are gonna be deciding by this line "bool isAligned = (matrixRows % tileDim == 0) && (matrixCols % tileDim
// == 0);"
template< int tileDim, int tileRowBlockSize, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionKernel( OutputMatrix resultMatrix,
                          const InputMatrix inputMatrix,
                          const Real matrixMultiplicator,
                          const Index gridIdx_x,
                          const Index gridIdx_y,
                          bool isAligned )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix.getColumns();
   const Index rows = inputMatrix.getRows();

   // Diagonal mapping of the CUDA blocks
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   // Read the tile to the shared memory
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   if( isAligned || readColumnPosition < columns ) {
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( isAligned || ( readRowPosition + rowBlock < rows ) ) {
            tile[ Backend::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix( readRowPosition + rowBlock, readColumnPosition );
         }
      }
   }
   __syncthreads();

   // Write the tile to the global memory
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   if( isAligned || writeColumnPosition < rows ) {
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( isAligned || ( writeRowPosition + rowBlock < columns ) ) {
            resultMatrix( writeRowPosition + rowBlock, writeColumnPosition ) =
               matrixMultiplicator * tile[ Backend::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ];
         }
      }
   }
#endif
}

// Kernel utilizing the Tensor Cores
template< typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
TensorCoreDenseMatrixTranspositionKernel( OutputMatrix resultMatrix,
                                          const InputMatrix inputMatrix,
                                          const Real matrixMultiplicator )
{
#ifdef __CUDACC__
   #ifdef USE_TENSOR_CORES
   using namespace nvcuda;

   const int WMMA_DIM = 16;  // WMMA tile dimensions

   const Index numRows = inputMatrix.getRows();
   const Index numCols = inputMatrix.getColumns();

   // Calculate the global indices of the element in the matrix
   Index globalRow = blockIdx.y * blockDim.y + threadIdx.y;
   Index globalCol = blockIdx.x * blockDim.x + threadIdx.x;

   // Define the WMMA tile for storing a fragment of the input and output matrices
   wmma::fragment< wmma::matrix_a, WMMA_DIM, WMMA_DIM, WMMA_DIM, half, wmma::col_major > input_frag;
   wmma::fragment< wmma::accumulator, WMMA_DIM, WMMA_DIM, WMMA_DIM, half > output_frag;

   // Ensure we are within bounds
   if( globalRow < numRows && globalCol < numCols ) {
      // Pointer to the element in the flat matrix data
      const half* inputPtr =
         reinterpret_cast< const half* >( inputMatrix.getValues().getData() + globalRow * numCols + globalCol );

      // Load the tile from the input matrix
      wmma::load_matrix_sync( input_frag, inputPtr, numCols );

      // Transpose the fragment simply by reinterpreting the input fragment as output
      for( int i = 0; i < input_frag.num_elements; ++i ) {
         output_frag.x[ i ] = input_frag.x[ i ];  // Direct copy, element-wise
      }

      // Calculate the output pointer based on transposed indices
      half* outputPtr = reinterpret_cast< half* >( resultMatrix.getValues().getData() + globalCol * numRows + globalRow );

      // Store the transposed tile
      wmma::store_matrix_sync( outputPtr, output_frag, numRows, wmma::mem_col_major );
   }
   #endif
#endif
}
};  //namespace TNL::Benchmarks::DenseMatrices
