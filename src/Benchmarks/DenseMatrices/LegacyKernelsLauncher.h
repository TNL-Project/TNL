#pragma once

#include "DenseMatrixMultiplicationKernels.h"
#include "DenseMatrixTranspositionKernels.h"

namespace TNL::Benchmarks::DenseMatrices {

template< typename RealType, typename DeviceType, typename IndexType >
class LegacyKernelsLauncher
{
public:
   static void
   launchMatrixMultiplicationKernel1( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel2( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel3( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel4( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel5( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel6( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixMultiplicationKernel7( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
                                      Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix );

   static void
   launchMatrixTranspositionKernel1( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix,
                                     Matrices::DenseMatrix< RealType, DeviceType, IndexType >& outputMatrix );
   static void
   launchMatrixTranspositionKernel2( Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix,
                                     Matrices::DenseMatrix< RealType, DeviceType, IndexType >& outputMatrix );
};

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel1(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 16;
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   const IndexType rowTiles = roundUpDivision( matrix1Rows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrix2Columns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync( MultiplicationKernel1< tileDim,
                                                            cudaBlockRows,
                                                            decltype( resultMatrixView ),
                                                            decltype( matrix1View ),
                                                            decltype( matrix2View ) >,
                                     launch_config,
                                     resultMatrixView,
                                     matrix1View,
                                     matrix2View,
                                     1.0,
                                     gridIdx_x,
                                     gridIdx_y );
      }
   }
   cudaStreamSynchronize( launch_config.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel2(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 16;
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   const IndexType rowTiles = roundUpDivision( matrix1Rows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrix2Columns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync( MultiplicationKernel2< tileDim,
                                                            cudaBlockRows,
                                                            decltype( resultMatrixView ),
                                                            decltype( matrix1View ),
                                                            decltype( matrix2View ) >,
                                     launch_config,
                                     resultMatrixView,
                                     matrix1View,
                                     matrix2View,
                                     1.0,
                                     gridIdx_x,
                                     gridIdx_y );
      }
   }
   cudaStreamSynchronize( launch_config.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel3(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 16;
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   const IndexType rowTiles = roundUpDivision( matrix1Rows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrix2Columns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync( MultiplicationKernel3< tileDim,
                                                            cudaBlockRows,
                                                            decltype( resultMatrixView ),
                                                            decltype( matrix1View ),
                                                            decltype( matrix2View ) >,
                                     launch_config,
                                     resultMatrixView,
                                     matrix1View,
                                     matrix2View,
                                     1.0,
                                     gridIdx_x,
                                     gridIdx_y );
      }
   }
   cudaStreamSynchronize( launch_config.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel4(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 16;
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   const IndexType rowTiles = roundUpDivision( matrix1Rows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrix2Columns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync(
            MultiplicationKernel4< tileDim, decltype( resultMatrixView ), decltype( matrix1View ), decltype( matrix2View ) >,
            launch_config,
            resultMatrixView,
            matrix1View,
            matrix2View,
            1.0 );
      }
   }
   cudaStreamSynchronize( launch_config.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel5(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 16;
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   const IndexType rowTiles = roundUpDivision( matrix1Rows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrix2Columns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync(
            MultiplicationKernel5< tileDim, decltype( resultMatrixView ), decltype( matrix1View ), decltype( matrix2View ) >,
            launch_config,
            resultMatrixView,
            matrix1View,
            matrix2View,
            1.0 );
      }
   }
   cudaStreamSynchronize( launch_config.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel6(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileSize = 64;  // each thread block handles a 64x64 block of the result matrix
   Backend::LaunchConfiguration fermiLaunchConfig;
   fermiLaunchConfig.blockSize.x = 16;  // 16 threads per block dimension
   fermiLaunchConfig.blockSize.y = 16;  // each thread handles a 4x4 tile, so 16x16 threads cover 64x64 elements

   IndexType matrix1Rows = matrix1.getRows();
   IndexType matrix2Columns = matrix2.getColumns();

   // Adjusting grid dimensions to ensure complete coverage of the matrix dimensions
   fermiLaunchConfig.gridSize.x = ( matrix2Columns + tileSize - 1 ) / tileSize;
   fermiLaunchConfig.gridSize.y = ( matrix1Rows + tileSize - 1 ) / tileSize;

   auto resultMatrixView = resultMatrix.getView();
   auto matrix1View = matrix1.getConstView();
   auto matrix2View = matrix2.getConstView();

   Backend::launchKernelAsync(
      MultiplicationKernel6< decltype( resultMatrixView ), decltype( matrix1View ), decltype( matrix2View ) >,
      fermiLaunchConfig,
      resultMatrixView,
      matrix1View,
      matrix2View,
      1.0 );

   cudaStreamSynchronize( fermiLaunchConfig.stream );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixMultiplicationKernel7(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix1,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix2,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& resultMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   #ifdef USE_TENSOR_CORES

   Backend::LaunchConfiguration launch_config_tensor;
   const IndexType rowTilesTensor = ( matrix1Rows + 15 ) / 16;
   const IndexType colTilesTensor = ( matrix2Columns + 15 ) / 16;
   benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
      { { "IndexType type", TNL::getType< IndexType >() },
        { "device", device },
        { "algorithm", "TensorCores" },
        { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
        { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

   for( IndexType gridIdx_y = 0; gridIdx_y < rowTilesTensor; gridIdx_y++ ) {
      for( IndexType gridIdx_x = 0; gridIdx_x < colTilesTensor; gridIdx_x++ ) {
         IndexType currentBlockRows = 16;
         if( ( gridIdx_y + 1 ) * 16 > matrix1Rows )  // Adjust rows for the last grid
            currentBlockRows = matrix1Rows % 16;

         IndexType currentBlockCols = 16;
         if( ( gridIdx_x + 1 ) * 16 > matrix2Columns )  // Adjust columns for the last grid
            currentBlockCols = matrix2Columns % 16;

         launch_config_tensor.gridSize.x = currentBlockCols;
         launch_config_tensor.gridSize.y = currentBlockRows;

         auto resultMatrixView = resultMatrix.getView();
         auto matrix1View = matrix1.getConstView();
         auto matrix2View = matrix2.getConstView();

         Backend::launchKernelAsync(
            MultiplicationKernel7< decltype( resultMatrixView ), decltype( matrix1View ), decltype( matrix2View ) >,
            launch_config_tensor,
            resultMatrixView,
            matrix1View,
            matrix2View,
            1.0 );
      }
   }
   cudaStreamSynchronize( launch_config_tensor.stream );
   TNL_CHECK_CUDA_DEVICE;
   #endif
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixTranspositionKernel1(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& outputMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 32;  // Example tile dimension
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

   IndexType matrixRows = matrix.getRows();
   IndexType matrixColumns = matrix.getColumns();

   const IndexType rowTiles = roundUpDivision( matrixRows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrixColumns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );
   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         if( ( gridIdx_x < columnGrids - 1 || matrixColumns % tileDim == 0 )
             && ( gridIdx_y < rowGrids - 1 || matrixRows % tileDim == 0 ) )
         {
            auto outputMatrixView = outputMatrix.getView();
            auto denseMatrixView = matrix.getConstView();
            constexpr auto kernel = DenseTranspositionAlignedKernel< tileDim,
                                                                     cudaBlockRows,
                                                                     decltype( outputMatrixView ),
                                                                     decltype( denseMatrixView ),
                                                                     RealType,
                                                                     IndexType >;
            Backend::launchKernelAsync( kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y );
         }
         else {
            auto outputMatrixView = outputMatrix.getView();
            auto denseMatrixView = matrix.getConstView();
            constexpr auto kernel = DenseTranspositionNonAlignedKernel< tileDim,
                                                                        cudaBlockRows,
                                                                        decltype( outputMatrixView ),
                                                                        decltype( denseMatrixView ),
                                                                        RealType,
                                                                        IndexType >;
            Backend::launchKernelAsync( kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y );
         }
      }
   Backend::streamSynchronize( launch_config.stream );
#endif
}

template< typename RealType, typename DeviceType, typename IndexType >
void
LegacyKernelsLauncher< RealType, DeviceType, IndexType >::launchMatrixTranspositionKernel2(
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& matrix,
   Matrices::DenseMatrix< RealType, DeviceType, IndexType >& outputMatrix )
{
#if defined( __CUDACC__ ) || ( __HIP__ )
   constexpr IndexType tileDim = 32;  // Example tile dimension
   constexpr IndexType matrixProductCudaBlockSize = 256;
   constexpr IndexType cudaBlockRows = matrixProductCudaBlockSize / tileDim;
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = tileDim;
   launch_config.blockSize.y = cudaBlockRows;
   launch_config.dynamicSharedMemorySize = tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

   IndexType matrixRows = matrix.getRows();
   IndexType matrixColumns = matrix.getColumns();

   const IndexType rowTiles = roundUpDivision( matrixRows, tileDim );
   const IndexType columnTiles = roundUpDivision( matrixColumns, tileDim );
   const IndexType rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
   const IndexType columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );
   for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
      for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         launch_config.gridSize.y = Backend::getMaxGridYSize();
         if( gridIdx_x == columnGrids - 1 )
            launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
         if( gridIdx_y == rowGrids - 1 )
            launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

         // Determine if this particular segment of the matrix is aligned
         bool isAligned = ( gridIdx_x < columnGrids - 1 || matrixColumns % tileDim == 0 )
                       && ( gridIdx_y < rowGrids - 1 || matrixRows % tileDim == 0 );

         auto outputMatrixView = outputMatrix.getView();
         auto denseMatrixView = matrix.getConstView();
         constexpr auto kernel = TranspositionKernel2< tileDim,
                                                       cudaBlockRows,
                                                       decltype( outputMatrixView ),
                                                       decltype( denseMatrixView ),
                                                       RealType,
                                                       IndexType >;

         // Launch the unified kernel with the isAligned parameter
         Backend::launchKernelAsync(
            kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y, isAligned );
      }
   Backend::streamSynchronize( launch_config.stream );
#endif
}

}  //namespace TNL::Benchmarks::DenseMatrices
