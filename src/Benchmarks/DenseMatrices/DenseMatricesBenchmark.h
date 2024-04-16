#pragma once

#include <TNL/Backend/Macros.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

#include "BlasBenchmark.h"

#include "DenseMatrixMultiplicationKernels.h"
#include "DenseMatrixTranspositionKernels.h"
#include "DenseMatricesResult.h"
#include <cmath>
#include <vector>

#if defined( __CUDACC__ )
   #include "CublasBenchmark.h"
   #include "CutlassBenchmark.h"
   #include "MagmaBenchmark.h"
#elif defined( __HIP__ )
   #include "HipBlasBenchmark.h"
#endif

namespace TNL::Benchmarks::DenseMatrices {
template< typename Real = double, typename Index = int >
struct DenseMatricesBenchmark
{
   using RealType = Real;
   using IndexType = Index;

#if defined( __CUDACC__ )
   using DeviceType = TNL::Devices::Cuda;
#endif

#if defined( __HIP__ )
   using DeviceType = TNL::Devices::Hip;
#endif

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "input-file", "Input file with dense matrices." );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-dense-matrices.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );
      config.addDelimiter( "Device settings:" );
      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "cuda" );
      config.addEntryEnum< TNL::String >( "host" );

#if defined( __CUDACC__ )
      config.addEntryEnum< TNL::String >( "cuda" );
      TNL::Devices::Cuda::configSetup( config );
#elif defined( __HIP__ )
      config.addEntryEnum< TNL::String >( "hip" );
      TNL::Devices::Hip::configSetup( config );
#endif

      config.addEntry< IndexType >( "loops", "Number of iterations for every computation.", 20 );
      config.addEntry< IndexType >( "verbose", "Verbose mode.", 1 );
      config.addEntry< TNL::String >( "fill-mode", "Method to fill matrices.", "linear" );
      config.addEntryEnum( "linear" );
      config.addEntryEnum( "trigonometric" );
   }

   TNL::Config::ParameterContainer parameters;
   DenseMatricesBenchmark( const TNL::Config::ParameterContainer& parameters_ ) : parameters( parameters_ ) {}

   bool
   runBenchmark()
   {
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const IndexType loops = parameters.getParameter< IndexType >( "loops" );
      const IndexType verbose = parameters.getParameter< IndexType >( "verbose" );
      bool isLinearFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      TNL::String device = parameters.getParameter< TNL::String >( "device" );

      std::cout << "Dense Matrices benchmark with " << TNL::getType< Real >() << " precision and device: " << device
                << std::endl;
      std::cout << std::endl;
      std::cout << "=== Dense Matrices Multiplication "
                   "==========================================================================================================="
                   "==================="
                << std::endl;
      std::cout << std::endl;

      const IndexType numMatrices = 1000;  // Number of matrices for the cycle
      IndexType matrix1Rows = 10;          // Number of rows in matrix1
      IndexType matrix1Columns = 10;       // Number of columns in matrix1 && rows in matrix2
      IndexType matrix2Columns = 10;       // Number of columns in matrix2

      for( IndexType i = 0; i < numMatrices; ++i ) {
         // Modify the matrix sizes for each iteration
         matrix1Rows += 100;
         matrix1Columns += 1;
         matrix2Columns += 300;

         if( device == "cuda" || device == "hip" || device == "all" ) {
#if defined( __CUDACC__ ) || ( __HIP__ )
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows, matrix1Columns );
            auto denseMatrix1View = denseMatrix1.getView();
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix2;
            denseMatrix2.setDimensions( matrix1Columns, matrix2Columns );
            auto denseMatrix2View = denseMatrix2.getView();

            // Fill the matrices
            const RealType h_x = 1.0 / 100;
            const RealType h_y = 1.0 / 100;

            auto fill1 = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix1Columns; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix1View.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix1Columns, fill1 );

            auto fill2 = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix2Columns; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 2 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix2View.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix2Columns, fill2 );

            // Create result matrices
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > resultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > cuBLASResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > CutlassResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > MagmaResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > BlasResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > HipBlasResultMatrix;

            resultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            cuBLASResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            CutlassResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            MagmaResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            BlasResultMatrix.setDimensions( matrix1Rows, matrix2Columns );

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "cuBLAS" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkcuBlas = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCuBLAS( denseMatrix1, denseMatrix2, cuBLASResultMatrix, false, false );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkcuBlas );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Magma" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function to perform matrix multiplication using MAGMA
            auto matrixMultiplicationBenchmarkMagma = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationMAGMA( denseMatrix1, denseMatrix2, MagmaResultMatrix, false, false );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesMAGMA = {
               cuBLASResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > MagmaResult( MagmaResultMatrix, benchmarkMatricesMAGMA );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkMagma, MagmaResult );
      #endif  //HAVE_MAGMA

      #ifdef HAVE_CUTLASS
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Cutlass" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function to perform matrix multiplication using Cutlass
            auto matrixMultiplicationBenchmarkCutlass = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCutlass( denseMatrix1, denseMatrix2, CutlassResultMatrix );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesCutlass = {
               cuBLASResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > CutlassResult( CutlassResultMatrix,
                                                                                  benchmarkMatricesCutlass );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkCutlass, CutlassResult );
      #endif  //HAVE_CUTLASS

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "TNL" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            constexpr Index tileDim = 16;  // Example tile dimension
            constexpr Index matrixProductCudaBlockSize = 256;
            constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = tileDim;
            launch_config.blockSize.y = cudaBlockRows;
            launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

            const Index rowTiles = roundUpDivision( matrix1Rows, tileDim );
            const Index columnTiles = roundUpDivision( matrix2Columns, tileDim );
            const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
            const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

            // Lambda function for the first kernel launch
            auto matrixMultiplicationBenchmarkOriginal = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( DenseMatrixProductKernel< tileDim,
                                                                           cudaBlockRows,
                                                                           decltype( resultMatrixView ),
                                                                           decltype( denseMatrix1View ),
                                                                           decltype( denseMatrix2View ) >,
                                                 launch_config,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0,
                                                 gridIdx_x,
                                                 gridIdx_y );
                  }
               }
               cudaStreamSynchronize( launch_config.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTNL = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > TNLResult( resultMatrix, benchmarkMatricesTNL );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkOriginal, TNLResult );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "TNL2" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkOptimized = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( OptimizedDenseMatrixProductKernel< tileDim,
                                                                                    cudaBlockRows,
                                                                                    decltype( resultMatrixView ),
                                                                                    decltype( denseMatrix1View ),
                                                                                    decltype( denseMatrix2View ) >,
                                                 launch_config,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0,
                                                 gridIdx_x,
                                                 gridIdx_y );
                  }
               }
               cudaStreamSynchronize( launch_config.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTNL2 = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > TNL2Result( resultMatrix, benchmarkMatricesTNL2 );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkOptimized, TNL2Result );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "2D SMA" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function for the optimized kernel 2 launch
            auto matrixMultiplicationBenchmarkOptimized2 = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( Optimized2DenseMatrixProductKernel< tileDim,
                                                                                     cudaBlockRows,
                                                                                     decltype( resultMatrixView ),
                                                                                     decltype( denseMatrix1View ),
                                                                                     decltype( denseMatrix2View ) >,
                                                 launch_config,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0,
                                                 gridIdx_x,
                                                 gridIdx_y );
                  }
               }
               cudaStreamSynchronize( launch_config.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesSMA = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > SMAResult( resultMatrix, benchmarkMatricesSMA );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkOptimized2, SMAResult );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Warptiling" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkWarptiling = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( WarpTilingDenseMatrixProductKernel< tileDim,
                                                                                     decltype( resultMatrixView ),
                                                                                     decltype( denseMatrix1View ),
                                                                                     decltype( denseMatrix2View ) >,
                                                 launch_config,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0 );
                  }
               }
               cudaStreamSynchronize( launch_config.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesWarptiling = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > WarptilingResult( resultMatrix,
                                                                                     benchmarkMatricesWarptiling );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkWarptiling, WarptilingResult );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Warptiling2" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkWarptiling2 = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( OptimizedWarpTilingDenseMatrixProductKernel< tileDim,
                                                                                              decltype( resultMatrixView ),
                                                                                              decltype( denseMatrix1View ),
                                                                                              decltype( denseMatrix2View ) >,
                                                 launch_config,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0 );
                  }
               }
               cudaStreamSynchronize( launch_config.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesWarptiling2 = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > Warptiling2Result( resultMatrix,
                                                                                      benchmarkMatricesWarptiling2 );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkWarptiling2, Warptiling2Result );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Fermi" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            // Fermi kernel specific launch configuration
            Backend::LaunchConfiguration fermiLaunchConfig;
            fermiLaunchConfig.blockSize.x = 16;  // Adjusted for the Fermi kernel
            fermiLaunchConfig.blockSize.y = 16;  // Adjusted for the Fermi kernel
            // fermiLaunchConfig.dynamicSharedMemorySize = sizeof( Real ) * ( ( 64 * 16 ) + ( 16 * 64 ) );

            auto matrixMultiplicationBenchmarkFermi = [ & ]() mutable
            {
               Index blockSize = 64;  // Each block computes a 64x64 block of the output matrix
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ ) {
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     fermiLaunchConfig.gridSize.x = ( matrix2Columns + blockSize - 1 ) / blockSize;
                     fermiLaunchConfig.gridSize.y = ( matrix1Rows + blockSize - 1 ) / blockSize;

                     // Adjust grid dimensions for potentially partial blocks on matrix edges
                     if( gridIdx_x == columnGrids - 1 )
                        fermiLaunchConfig.gridSize.x =
                           ( ( columnTiles % blockSize == 0 ) ? ( columnTiles / blockSize ) : ( columnTiles / blockSize + 1 ) );
                     if( gridIdx_y == rowGrids - 1 )
                        fermiLaunchConfig.gridSize.y =
                           ( ( rowTiles % blockSize == 0 ) ? ( rowTiles / blockSize ) : ( rowTiles / blockSize + 1 ) );

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( optimizedFermiGemmKernel< decltype( resultMatrixView ),
                                                                           decltype( denseMatrix1View ),
                                                                           decltype( denseMatrix2View ) >,
                                                 fermiLaunchConfig,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0 );
                  }
               }
               cudaStreamSynchronize( fermiLaunchConfig.stream );
               TNL_CHECK_CUDA_DEVICE;
               /*
               std::cout << "matrix1: " << denseMatrix1 << std::endl;
               std::cout << "matrix2: " << denseMatrix2 << std::endl;
               std::cout << "result: " << resultMatrix << std::endl;
               std::cout << "result should be : " << cuBLASResultMatrix << std::endl;
               */
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesFermi = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > FermiResult( resultMatrix, benchmarkMatricesFermi );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkFermi, FermiResult );

      #ifdef USE_TENSOR_CORES
            Backend::LaunchConfiguration launch_config_tensor;
            const int rowTilesTensor = ( matrix1Rows + 15 ) / 16;
            const int colTilesTensor = ( matrix2Columns + 15 ) / 16;
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "TensorCores" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkTensorCores = [ & ]() mutable
            {
               for( int gridIdx_y = 0; gridIdx_y < rowTilesTensor; gridIdx_y++ ) {
                  for( int gridIdx_x = 0; gridIdx_x < colTilesTensor; gridIdx_x++ ) {
                     int currentBlockRows = 16;
                     if( ( gridIdx_y + 1 ) * 16 > matrix1Rows )  // Adjust rows for the last grid
                        currentBlockRows = matrix1Rows % 16;

                     int currentBlockCols = 16;
                     if( ( gridIdx_x + 1 ) * 16 > matrix2Columns )  // Adjust columns for the last grid
                        currentBlockCols = matrix2Columns % 16;

                     launch_config_tensor.gridSize.x = currentBlockCols;
                     launch_config_tensor.gridSize.y = currentBlockRows;

                     auto resultMatrixView = resultMatrix.getView();
                     auto denseMatrix1View = denseMatrix1.getConstView();
                     auto denseMatrix2View = denseMatrix2.getConstView();

                     Backend::launchKernelAsync( TensorCoreDenseMatrixProductKernel< decltype( resultMatrixView ),
                                                                                     decltype( denseMatrix1View ),
                                                                                     decltype( denseMatrix2View ) >,
                                                 launch_config_tensor,
                                                 resultMatrixView,
                                                 denseMatrix1View,
                                                 denseMatrix2View,
                                                 1.0 );
                  }
               }
               cudaStreamSynchronize( launch_config_tensor.stream );
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTensorCores = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };

            DenseMatricesResult< RealType, DeviceType, IndexType > TensorCoresResult( resultMatrix,
                                                                                      benchmarkMatricesTensorCores );

            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkTensorCores, TensorCoresResult );
      #endif  // USE_TENSOR_CORES
   #endif
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "Final" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkFinal = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1, denseMatrix2 );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesFinal = {
               cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > FinalResult( resultMatrix, benchmarkMatricesFinal );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkFinal, FinalResult );

   #if defined( __HIP__ )

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "HipBlas" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkHIPBLAS = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationHIPBLAS( denseMatrix1, denseMatrix2, HipBlasResultMatrix, false, false );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkHIPBLAS );

   #endif

            std::cout << "-----------------------------------------------------------------------------------------------------"
                         "-----------------------------------------------------------"
                      << std::endl;
#endif  // ( __CUDACC__ ) || defined( __HIP__ )
         }

         if( device == "host" || device == "all" ) {
            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows, matrix1Columns );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > denseMatrix2;
            denseMatrix2.setDimensions( matrix1Columns, matrix2Columns );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > resultMatrix;
            resultMatrix.setDimensions( matrix1Rows, matrix2Columns );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > BlasResultMatrix;
            BlasResultMatrix.setDimensions( matrix1Rows, matrix2Columns );

            const RealType h_x = 1.0 / 100;
            const RealType h_y = 1.0 / 100;

            for( int i = 0; i < matrix1Rows; i++ ) {
               for( int j = 0; j < matrix1Columns; j++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix1.setElement( i, j, value );
               }
            }

            for( int i = 0; i < matrix1Columns; i++ ) {
               for( int j = 0; j < matrix2Columns; j++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix2.setElement( i, j, value );
               }
            }

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "BLAS" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

#ifdef HAVE_BLAS
            auto matrixMultiplicationBenchmarkBLAS = [ & ]() mutable
            {
               matrixMultiplicationBLAS( denseMatrix1, denseMatrix2, BlasResultMatrix );
            };
            benchmark.time< Devices::Host >( device, matrixMultiplicationBenchmarkBLAS );
#endif  //HAVE_BLAS

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "TNL" },
                 { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                 { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkTNL = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1, denseMatrix2, 1.0 );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > > benchmarkMatricesCPU = {
               BlasResultMatrix
            };
            DenseMatricesResult< RealType, Devices::Host, IndexType > CPUResult( resultMatrix, benchmarkMatricesCPU );
            benchmark.time< Devices::Host >( device, matrixMultiplicationBenchmarkTNL, CPUResult );

            std::cout << "-----------------------------------------------------------------------------------------------------"
                         "-----------------------------------------------------------"
                      << std::endl;
         }
      }

      std::cout << std::endl;
      std::cout << "=== Dense Matrix Trasnposition "
                   "==========================================================================================================="
                   "========"
                << std::endl;
      std::cout << std::endl;

      IndexType dmatrix1Rows = 10;     // Number of rows in matrix1 (same as columns in matrix2)
      IndexType dmatrix1Columns = 10;  // Number of columns in matrix1
      IndexType numMatrices1 = 10;     // NUmber of matrices that are going to be generated
      for( IndexType i = 0; i < numMatrices1; ++i ) {
         // Modify the matrix sizes for each iteration
         dmatrix1Rows += 100;
         dmatrix1Columns += 100;

         if( device == "cuda" || device == "hip" || device == "all" ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "MAGMA" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

#if defined( __CUDACC__ ) || ( __HIP__ )
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix;
            denseMatrix.setDimensions( dmatrix1Rows, dmatrix1Columns );
            auto denseMatrixView = denseMatrix.getView();

            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > outputMatrix;
            outputMatrix.setDimensions( dmatrix1Columns, dmatrix1Rows );

            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > MagmaOutputMatrix;
            MagmaOutputMatrix.setDimensions( dmatrix1Columns, dmatrix1Rows );

            // Fill the matrix
            const RealType h_x = 1.0 / 100;
            const RealType h_y = 1.0 / 100;

            auto fill = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < dmatrix1Columns; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrixView.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, dmatrix1Columns, fill );
   #if defined( __CUDACC__ )
      #ifdef HAVE_MAGMA
            // Lambda function to perform matrix transposition using MAGMA
            auto matrixTranspositionBenchmarkMagma = [ & ]() mutable
            {
               denseMatrixTransposeMAGMA( denseMatrix, MagmaOutputMatrix );
            };
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkMagma );
      #endif  //HAVE_MAGMA
   #endif

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "TNL" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            constexpr Index tileDim = 32;  // Example tile dimension
            constexpr Index matrixProductCudaBlockSize = 256;
            constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = tileDim;
            launch_config.blockSize.y = cudaBlockRows;
            launch_config.dynamicSharedMemorySize =
               tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

            const Index rowTiles = roundUpDivision( dmatrix1Rows, tileDim );
            const Index columnTiles = roundUpDivision( dmatrix1Columns, tileDim );
            const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
            const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

            auto matrixTranspositionBenchmarkTNL = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     if( ( gridIdx_x < columnGrids - 1 || denseMatrix.getColumns() % tileDim == 0 )
                         && ( gridIdx_y < rowGrids - 1 || denseMatrix.getRows() % tileDim == 0 ) )
                     {
                        auto outputMatrixView = outputMatrix.getView();
                        auto denseMatrixView = denseMatrix.getConstView();
                        constexpr auto kernel = DenseTranspositionAlignedKernel< tileDim,
                                                                                 cudaBlockRows,
                                                                                 decltype( outputMatrixView ),
                                                                                 decltype( denseMatrixView ),
                                                                                 Real,
                                                                                 Index >;
                        Backend::launchKernelAsync(
                           kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y );
                     }
                     else {
                        auto outputMatrixView = outputMatrix.getView();
                        auto denseMatrixView = denseMatrix.getConstView();
                        constexpr auto kernel = DenseTranspositionNonAlignedKernel< tileDim,
                                                                                    cudaBlockRows,
                                                                                    decltype( outputMatrixView ),
                                                                                    decltype( denseMatrixView ),
                                                                                    Real,
                                                                                    Index >;
                        Backend::launchKernelAsync(
                           kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y );
                     }
                  }
               Backend::streamSynchronize( launch_config.stream );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTransposition = {
               MagmaOutputMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult( outputMatrix,
                                                                                        benchmarkMatricesTransposition );
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkTNL, TranspositionResult );

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "Combined" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            auto matrixTranspositionBenchmarkCombined = [ & ]() mutable
            {
               for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
                  for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
                     launch_config.gridSize.x = Backend::getMaxGridXSize();
                     launch_config.gridSize.y = Backend::getMaxGridYSize();
                     if( gridIdx_x == columnGrids - 1 )
                        launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                     if( gridIdx_y == rowGrids - 1 )
                        launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                     // Determine if this particular segment of the matrix is aligned
                     bool isAligned = ( gridIdx_x < columnGrids - 1 || denseMatrix.getColumns() % tileDim == 0 )
                                   && ( gridIdx_y < rowGrids - 1 || denseMatrix.getRows() % tileDim == 0 );

                     auto outputMatrixView = outputMatrix.getView();
                     auto denseMatrixView = denseMatrix.getConstView();
                     constexpr auto kernel = DenseTranspositionKernel< tileDim,
                                                                       cudaBlockRows,
                                                                       decltype( outputMatrixView ),
                                                                       decltype( denseMatrixView ),
                                                                       Real,
                                                                       Index >;

                     // Launch the unified kernel with the isAligned parameter
                     Backend::launchKernelAsync(
                        kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y, isAligned );
                  }
               Backend::streamSynchronize( launch_config.stream );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > >
               benchmarkMatricesTranspositionCombined = { MagmaOutputMatrix };
            DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult2(
               outputMatrix, benchmarkMatricesTranspositionCombined );
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkCombined, TranspositionResult2 );

   #ifdef USE_TENSOR_CORES
            Backend::LaunchConfiguration launch_config_tensor;
            const int rowTilesTensor = ( matrix1Rows + 15 ) / 16;
            const int colTilesTensor = ( matrix2Columns + 15 ) / 16;

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "TensorCores" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            auto matrixTranspositionBenchmarkTensorCores = [ & ]() mutable
            {
               for( int gridIdx_y = 0; gridIdx_y < rowTilesTensor; gridIdx_y++ )
                  for( int gridIdx_x = 0; gridIdx_x < colTilesTensor; gridIdx_x++ ) {
                     int currentBlockRows = 16;
                     if( ( gridIdx_y + 1 ) * 16 > matrix1Rows )  // Adjust rows for the last grid
                        currentBlockRows = matrix1Rows % 16;

                     int currentBlockCols = 16;
                     if( ( gridIdx_x + 1 ) * 16 > matrix2Columns )  // Adjust columns for the last grid
                        currentBlockCols = matrix2Columns % 16;

                     launch_config_tensor.gridSize.x = currentBlockCols;
                     launch_config_tensor.gridSize.y = currentBlockRows;

                     auto outputMatrixView = outputMatrix.getView();
                     auto denseMatrixView = denseMatrix.getConstView();
                     constexpr auto kernel = TensorCoreDenseMatrixTranspositionKernel< decltype( outputMatrixView ),
                                                                                       decltype( denseMatrixView ),
                                                                                       Real,
                                                                                       Index >;

                     Backend::launchKernelAsync( kernel, launch_config_tensor, outputMatrixView, denseMatrixView, 1.0 );
                  }
               Backend::streamSynchronize( launch_config_tensor.stream );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > >
               benchmarkMatricesTranspositionTensorCores = { MagmaOutputMatrix };
            DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResultTensorCores(
               outputMatrix, benchmarkMatricesTranspositionTensorCores );
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkTensorCores, TranspositionResultTensorCores );
   #endif  // USE_TENSOR_CORES

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "Final" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkFinal = [ & ]() mutable
            {
               outputMatrix.getTransposition( denseMatrix );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTranspositionFinal = {
               MagmaOutputMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult3( outputMatrix,
                                                                                         benchmarkMatricesTranspositionFinal );
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkFinal, TranspositionResult3 );

            if( dmatrix1Rows == dmatrix1Columns ) {
               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                  { "index type", TNL::getType< Index >() },
                  { "device", device },
                  { "algorithm", "InPlace" },
                  { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
               } ) );

               // Lambda function to perform matrix transposition using TNL
               auto matrixTranspositionBenchmarkInPlace = [ & ]() mutable
               {
                  denseMatrix.getInPlaceTransposition();
               };
               std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > >
                  benchmarkMatricesTranspositionInPlace = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult4(
                  denseMatrix, benchmarkMatricesTranspositionInPlace );
               benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkInPlace, TranspositionResult3 );
            }

            std::cout << "-----------------------------------------------------------------------------------------------------"
                         "----------------------------------------------"
                      << std::endl;
#endif  // ( __CUDACC__ ) || defined( __HIP__ )
         }
         if( device == "host" || device == "all" ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "device", device },
               { "algorithm", "TNL" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > denseMatrix;
            denseMatrix.setDimensions( dmatrix1Rows, dmatrix1Columns );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > outputMatrix;
            outputMatrix.setDimensions( dmatrix1Columns, dmatrix1Rows );

            const RealType h_x = 1.0 / 100;
            const RealType h_y = 1.0 / 100;

            for( int i = 0; i < dmatrix1Rows; i++ ) {
               for( int j = 0; j < dmatrix1Columns; j++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix.setElement( i, j, value );
               }
            }

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkTNL = [ & ]() mutable
            {
               outputMatrix.getTransposition( denseMatrix );
            };
            benchmark.time< Devices::Host >( device, matrixTranspositionBenchmarkTNL );

            std::cout << "-----------------------------------------------------------------------------------------------------"
                         "----------------------------------------------"
                      << std::endl;
         }
      }
      std::cout << std::endl;
      std::cout << "=== Final Kernel Tests "
                   "==========================================================================================================="
                   "========"
                << std::endl;
      std::cout << std::endl;

      const IndexType numMatrices2 = 100;  // Number of matrices for the cycle
      IndexType matrix1Rows2 = 10;         // Number of rows in matrix1
      IndexType matrix1Columns2 = 10;      // Number of columns in matrix1 && rows in matrix2
      IndexType matrix2Columns2 = 10;      // Number of columns in matrix2

      for( IndexType i = 0; i < numMatrices2; ++i ) {
         // Modify the matrix sizes for each iteration
         matrix1Rows2 += 10;
         matrix1Columns2 += 20;
         matrix2Columns2 += 30;

         // Multiplication with TransposeState
         if( device == "cuda" || device == "hip" || device == "all" ) {
#if defined( __CUDACC__ ) || ( __HIP__ )

            // Original Matrices
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows2, matrix1Columns2 );
            auto denseMatrix1View = denseMatrix1.getView();

            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix2;
            denseMatrix2.setDimensions( matrix1Columns2, matrix2Columns2 );  // Matches inner dimension of Matrix1
            auto denseMatrix2View = denseMatrix2.getView();

            // Transposed Matrix1 (For Transpose A Only and Transpose Both Matrices)
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix1Transposed;
            denseMatrix1Transposed.setDimensions( matrix1Columns2, matrix1Rows2 );
            auto denseMatrix1TransposedView = denseMatrix1Transposed.getView();

            // Transposed Matrix2 (For Transpose B Only and Transpose Both Matrices)
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix2Transposed;
            denseMatrix2Transposed.setDimensions( matrix2Columns2,
                                                  matrix1Columns2 );  // For matching with transposed dimensions of Matrix1
            auto denseMatrix2TransposedView = denseMatrix2Transposed.getView();

            // Fill the matrices
            const RealType h_x = 1.0 / 100;
            const RealType h_y = 1.0 / 100;

            auto fill1 = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix1Columns2; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 3 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix1View.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix1Columns2, fill1 );

            auto fill2 = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix2Columns2; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 2 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix2View.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix2Columns2, fill2 );

            auto fill1Transposed = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix1Rows2; i++ ) {  // Note: Iterating over `matrix1Rows` for the transposed matrix
                  RealType value;
                  if( isLinearFill ) {
                     value = 2 + i * 2;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix1TransposedView.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix1Rows2, fill1Transposed );

            auto fill2Transposed = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
            {
               for( IndexType i = 0; i < matrix1Columns2; i++ ) {
                  RealType value;
                  if( isLinearFill ) {
                     value = 2 * i;
                  }
                  else {  // trigonometric
                     value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
                  }
                  denseMatrix2TransposedView.setElement( i, rowIdx, value );
               }
            };
            TNL::Algorithms::parallelFor< DeviceType >( 0, matrix1Columns2, fill2Transposed );

            // Create result matrices
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > resultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > MagmaResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > CuBLASResultMatrix;
            TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > HipBlasResultMatrix;

            resultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            MagmaResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            CuBLASResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            HipBlasResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );

            std::cout << std::endl;
            std::cout << "=== A Transposed "
                         "====================================================================================================="
                         "=========================================="
                      << std::endl;
            std::cout << std::endl;

   #if defined( __HIP__ )
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "hipblasA" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix2Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using HipBlas
            auto matrixMultiplicationBenchmarkHipBlasTransB = [ & ]() mutable
            {
               // Call HipBLAS matrix multiplication function for both matrices transposed
               matrixMultiplicationHIPBLAS( denseMatrix1Transposed, denseMatrix2, HipBlasResultMatrix, true, false );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkHipBlasTransB );
   #endif

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "cublasA" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix2Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using cuBLAS
            auto matrixMultiplicationBenchmarkCuBlasTransA = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function function for both matrices transposed
               matrixMultiplicationCuBLAS( denseMatrix1Transposed, denseMatrix2, CuBLASResultMatrix, true, false );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkCuBlasTransA );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "magmaA" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix2Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using MAGMA
            auto matrixMultiplicationBenchmarkMagmaTransA = [ & ]() mutable
            {
               // Call MAGMA matrix multiplication function for A transposed
               matrixMultiplicationMAGMA( denseMatrix1Transposed, denseMatrix2, MagmaResultMatrix, true, false );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkMagmaTransA );
      #endif  //HAVE_MAGMA

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "tnlA" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix2Columns2 ) } } ) );

            auto matrixMultiplicationBenchmarkTransA = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1Transposed,
                                              denseMatrix2,
                                              1.0,
                                              TNL::Matrices::TransposeState::Transpose,
                                              TNL::Matrices::TransposeState::None );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesATrans = {
               CuBLASResultMatrix, MagmaResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > ATransResult( resultMatrix, benchmarkMatricesATrans );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkTransA, ATransResult );
   #endif
            std::cout << std::endl;
            std::cout << "=== B Transposed "
                         "====================================================================================================="
                         "=========================================="

                      << std::endl;
            std::cout << std::endl;

   #if defined( __HIP__ )

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "hipblasB" },
                 { "matrix1 size", std::to_string( matrix1Rows2 ) + "x" + std::to_string( matrix1Columns2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using HipBlas
            auto matrixMultiplicationBenchmarkHipBlasTransA = [ & ]() mutable
            {
               // Call HipBLAS matrix multiplication function for both matrices transposed
               matrixMultiplicationHIPBLAS( denseMatrix1, denseMatrix2Transposed, HipBlasResultMatrix, false, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkHipBlasTransA );
   #endif
   #if defined( __CUDACC__ )

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "cublasB" },
                 { "matrix1 size", std::to_string( matrix1Rows2 ) + "x" + std::to_string( matrix1Columns2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using cuBLAS
            auto matrixMultiplicationBenchmarkCuBlasTransB = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function for B transposed
               matrixMultiplicationCuBLAS( denseMatrix1, denseMatrix2Transposed, CuBLASResultMatrix, false, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkCuBlasTransB );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "magmaB" },
                 { "matrix1 size", std::to_string( matrix1Rows2 ) + "x" + std::to_string( matrix1Columns2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using MAGMA
            auto matrixMultiplicationBenchmarkMagmaTransB = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function for B transposed
               matrixMultiplicationMAGMA( denseMatrix1, denseMatrix2Transposed, MagmaResultMatrix, false, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkMagmaTransB );

      #endif  //HAVE_MAGMA

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "tnlB" },
                 { "matrix1 size", std::to_string( matrix1Rows2 ) + "x" + std::to_string( matrix1Columns2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            auto matrixMultiplicationBenchmarkTransB = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1,
                                              denseMatrix2Transposed,
                                              1.0,
                                              TNL::Matrices::TransposeState::None,
                                              TNL::Matrices::TransposeState::Transpose );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesBTrans = {
               CuBLASResultMatrix, MagmaResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > BTransResult( resultMatrix, benchmarkMatricesBTrans );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkTransB, BTransResult );
   #endif
            std::cout << std::endl;
            std::cout << "=== A and B Transposed "
                         "====================================================================================================="
                         "===================================="

                      << std::endl;
            std::cout << std::endl;

   #if defined( __HIP__ )

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "hipblasAB" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using HipBlas
            auto matrixMultiplicationBenchmarkHipBlasTransBoth = [ & ]() mutable
            {
               // Call HipBLAS matrix multiplication function for both matrices transposed
               matrixMultiplicationHIPBLAS( denseMatrix1Transposed, denseMatrix2Transposed, HipBlasResultMatrix, true, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkHipBlasTransBoth );
   #endif
   #if defined( __CUDACC__ )

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "cublasAB" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using cuBLAS
            auto matrixMultiplicationBenchmarkCuBlasTransBoth = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function for both matrices transposed
               matrixMultiplicationCuBLAS( denseMatrix1Transposed, denseMatrix2Transposed, CuBLASResultMatrix, true, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkCuBlasTransBoth );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "magmaAB" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            // Lambda function to perform matrix multiplication using MAGMA
            auto matrixMultiplicationBenchmarkMagmaTransBoth = [ & ]() mutable
            {
               // Call MAGMA matrix multiplication function for both matrices transposed
               matrixMultiplicationMAGMA( denseMatrix1Transposed, denseMatrix2Transposed, MagmaResultMatrix, true, true );
            };
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkMagmaTransBoth );
      #endif  //HAVE_MAGMA

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns(
               { { "index type", TNL::getType< Index >() },
                 { "device", device },
                 { "algorithm", "tnlAB" },
                 { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                 { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            auto matrixMultiplicationBenchmarkTransBoth = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1Transposed,
                                              denseMatrix2Transposed,
                                              1.0,
                                              TNL::Matrices::TransposeState::Transpose,
                                              TNL::Matrices::TransposeState::Transpose );
            };
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesBothTrans = {
               CuBLASResultMatrix, MagmaResultMatrix
            };
            DenseMatricesResult< RealType, DeviceType, IndexType > BothTransResult( resultMatrix, benchmarkMatricesBothTrans );
            benchmark.time< DeviceType >( device, matrixMultiplicationBenchmarkTransBoth, BothTransResult );
   #endif

#endif  // ( __CUDACC__ ) || defined( __HIP__ )
         }
      }
      return true;
   }
};
}  // namespace TNL::Benchmarks::DenseMatrices
