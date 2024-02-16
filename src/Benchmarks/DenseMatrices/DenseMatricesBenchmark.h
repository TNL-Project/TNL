#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/detail/ParallelFor2D.h>

#include "CublasBenchmark.h"
#include "BlasBenchmark.h"
#include "CutlassBenchmark.h"
#include "MagmaBenchmark.h"
#include "DenseMatrixMultiplicationKernels.h"
#include "DenseMatrixTranspositionKernels.h"
#include "DenseMatricesResult.h"
#include <cmath>
#include <vector>


namespace TNL::Benchmarks::DenseMatrices {

template <typename Real = double, typename Index = int>
struct DenseMatricesBenchmark {
   using RealType = Real;
   using IndexType = Index;

   static void configSetup(TNL::Config::ConfigDescription& config) {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("input-file", "Input file with dense matrices.");
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-dense-matrices.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");
      config.addDelimiter("Device settings:");
      config.addEntry<TNL::String>("device", "Device the computation will run on.", "cuda");
      config.addEntryEnum<TNL::String>("cuda");
      config.addEntryEnum<TNL::String>("host");
      TNL::Devices::Cuda::configSetup(config);
      config.addEntry<int>("loops", "Number of iterations for every computation.", 20);
      config.addEntry<int>("verbose", "Verbose mode.", 1);
    }

   TNL::Config::ParameterContainer parameters;
   DenseMatricesBenchmark(const TNL::Config::ParameterContainer& parameters_) : parameters(parameters_) {}

   bool runBenchmark() {
      const TNL::String logFileName = parameters.getParameter<TNL::String>("log-file");
      const TNL::String outputMode = parameters.getParameter<TNL::String>("output-mode");
      const int loops = parameters.getParameter<int>("loops");
      const int verbose = parameters.getParameter<int>("verbose");

      auto mode = std::ios::out;
      if (outputMode == "append")
         mode |= std::ios::app;
      std::ofstream logFile(logFileName.getString(), mode);
      TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

      std::map<std::string, std::string> metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson(metadata, logFileName, ".metadata.json");

      TNL::String device = parameters.getParameter<TNL::String>("device");

      std::cout << "Dense Matrices benchmark with " << TNL::getType<Real>() << " precision and device: " << device << std::endl;
      std::cout << std::endl;
      std::cout << "=== Dense Matrices Multiplication ==============================================================================================================================" << std::endl;
      std::cout << std::endl;

      const int numMatrices = 10; // Number of matrices for the cycle
      int matrix1Rows = 20; // Number of rows in matrix1
      int matrix1Columns = 10; // Number of columns in matrix1 && rows in matrix2
      int matrix2Columns = 30; // Number of columns in matrix2

      for (int i = 0; i < numMatrices; ++i) {

         // Modify the matrix sizes for each iteration
         matrix1Rows += 10;
         matrix1Columns += 10;
         matrix2Columns += 10;

         if (device == "cuda" || device == "all") {

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "cuBLAS" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function to perform matrix multiplication using cuBLAS
#ifdef __CUDACC__

            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType > denseMatrix1;
            denseMatrix1.setDimensions(matrix1Rows, matrix1Columns);

            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType > denseMatrix2;
            denseMatrix2.setDimensions(matrix1Columns, matrix2Columns);

            // Fill the matrices
            //const double h_x = 1.0 / 100;
            //const double h_y = 1.0 / 100;

            for (int i = 0; i < matrix1Rows; i++) {
               for (int j = 0; j < matrix1Columns; j++) {
                  //double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  double value = 3*i;
                  denseMatrix1.setElement(i, j, value);
               }
            }

            for (int i = 0; i < matrix1Columns; i++) {
               for (int j = 0; j < matrix2Columns; j++) {
                  //double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  double value = 2*i;
                  denseMatrix2.setElement(i, j, value);
               }
            }

            // Create a result matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> resultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> cuBLASResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> CutlassResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> MagmaResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> BlasResultMatrix;

            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            cuBLASResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            CutlassResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            MagmaResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            BlasResultMatrix.setDimensions(matrix1Rows, matrix2Columns);

            auto matrixMultiplicationBenchmarkcuBlas = [&]() mutable {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCuBLAS(denseMatrix1, denseMatrix2, cuBLASResultMatrix);
            };
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkcuBlas);

#ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "Magma" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function to perform matrix multiplication using MAGMA
            auto matrixMultiplicationBenchmarkMagma = [&]() mutable {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationMAGMA(denseMatrix1, denseMatrix2, MagmaResultMatrix);
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesMAGMA = {cuBLASResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> MagmaResult(MagmaResultMatrix, benchmarkMatricesMAGMA);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkMagma, MagmaResult);
#endif //HAVE_MAGMA

#ifdef HAVE_CUTLASS
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "Cutlass" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function to perform matrix multiplication using Cutlass
            auto matrixMultiplicationBenchmarkCutlass = [&]() mutable {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCutlass(denseMatrix1, denseMatrix2, CutlassResultMatrix);
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesCutlass = {cuBLASResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> CutlassResult(CutlassResultMatrix, benchmarkMatricesCutlass);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkCutlass, CutlassResult);
#endif //HAVE_CUTLASS

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "TNL" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            constexpr Index tileDim = 16; // Example tile dimension
            constexpr Index matrixProductCudaBlockSize = 256;
            constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = tileDim;
            launch_config.blockSize.y = cudaBlockRows;
            launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

            const Index rowTiles = roundUpDivision(matrix1Rows, tileDim);
            const Index columnTiles = roundUpDivision(matrix2Columns, tileDim);
            const Index rowGrids = roundUpDivision(rowTiles, Backend::getMaxGridYSize());
            const Index columnGrids = roundUpDivision(columnTiles, Backend::getMaxGridXSize());

            // Lambda function for the first kernel launch
            auto matrixMultiplicationBenchmarkOriginal = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                     for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Backend::launchKernelAsync(DenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                launch_config,
                                                resultMatrixView,
                                                denseMatrix1View,
                                                denseMatrix2View,
                                                1.0,
                                                gridIdx_x,
                                                gridIdx_y);
                     }
               }
               cudaStreamSynchronize(launch_config.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesTNL = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> TNLResult(resultMatrix, benchmarkMatricesTNL );
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOriginal, TNLResult);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "TNL2" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkOptimized = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                     for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Backend::launchKernelAsync(OptimizedDenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                launch_config,
                                                resultMatrixView,
                                                denseMatrix1View,
                                                denseMatrix2View,
                                                1.0,
                                                gridIdx_x,
                                                gridIdx_y);
                     }
               }
               cudaStreamSynchronize(launch_config.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesTNL2 = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> TNL2Result(resultMatrix, benchmarkMatricesTNL2);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOptimized, TNL2Result);


            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "2D SMA" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function for the optimized kernel 2 launch
            auto matrixMultiplicationBenchmarkOptimized2 = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                     for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();


                        Backend::launchKernelAsync(Optimized2DenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                launch_config,
                                                resultMatrixView,
                                                denseMatrix1View,
                                                denseMatrix2View,
                                                1.0,
                                                gridIdx_x,
                                                gridIdx_y);
                     }
               }
               cudaStreamSynchronize(launch_config.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesSMA = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> SMAResult(resultMatrix, benchmarkMatricesSMA);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOptimized2,SMAResult);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "index type", TNL::getType<Index>() },
            { "device", device },
            { "algorithm", "Warptiling" },
            { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
            { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkWarptiling = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                  for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Backend::launchKernelAsync(WarpTilingDenseMatrixProductKernel<tileDim, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                launch_config,
                                                resultMatrixView,
                                                denseMatrix1View,
                                                denseMatrix2View,
                                                1.0);
                  }
               }
               cudaStreamSynchronize(launch_config.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesWarptiling = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> WarptilingResult(resultMatrix, benchmarkMatricesWarptiling);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkWarptiling, WarptilingResult);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "index type", TNL::getType<Index>() },
            { "device", device },
            { "algorithm", "Warptiling2" },
            { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
            { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkWarptiling2 = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                  for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Backend::launchKernelAsync(OptimizedWarpTilingDenseMatrixProductKernel<tileDim, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                launch_config,
                                                resultMatrixView,
                                                denseMatrix1View,
                                                denseMatrix2View,
                                                1.0);
                  }
               }
               cudaStreamSynchronize(launch_config.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesWarptiling2 = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> Warptiling2Result(resultMatrix, benchmarkMatricesWarptiling2);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkWarptiling2,Warptiling2Result);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "index type", TNL::getType<Index>() },
            { "device", device },
            { "algorithm", "Fermi" },
            { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
            { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Fermi kernel specific launch configuration
            Backend::LaunchConfiguration fermiLaunchConfig;
            fermiLaunchConfig.blockSize.x = 16; // Adjusted for the Fermi kernel
            fermiLaunchConfig.blockSize.y = 16; // Adjusted for the Fermi kernel
            fermiLaunchConfig.dynamicSharedMemorySize = sizeof(Real) * ((64 * 16) + (16 * 64));

            auto matrixMultiplicationBenchmarkFermi = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                  for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        Index blockSize = 64; // Each block computes a 64x64 block of the output matrix
                        fermiLaunchConfig.gridSize.x = (matrix2Columns + blockSize - 1) / blockSize;
                        fermiLaunchConfig.gridSize.y = (matrix1Rows + blockSize - 1) / blockSize;

                        // Adjust the last block size if the matrix dimensions are not multiples of the block size
                        if (gridIdx_x == columnGrids - 1)
                           fermiLaunchConfig.gridSize.x = (columnTiles % blockSize == 0) ? blockSize : columnTiles % blockSize;
                        if (gridIdx_y == rowGrids - 1)
                           fermiLaunchConfig.gridSize.y = (rowTiles % blockSize == 0) ? blockSize : rowTiles % blockSize;

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Backend::launchKernelAsync(optimizedFermiGemmKernel<decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
                                                   fermiLaunchConfig,
                                                   resultMatrixView,
                                                   denseMatrix1View,
                                                   denseMatrix2View,
                                                   1.0);
                  }
               }
               cudaStreamSynchronize(fermiLaunchConfig.stream);
               TNL_CHECK_CUDA_DEVICE;
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesFermi = {cuBLASResultMatrix, MagmaResultMatrix, CutlassResultMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> FermiResult(resultMatrix, benchmarkMatricesFermi);
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkFermi, FermiResult);

         std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
#endif //__CUDACC__
      }

         if (device == "host" || device == "all") {

            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix1;
            denseMatrix1.setDimensions(matrix1Rows, matrix1Columns);

            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix2;
            denseMatrix2.setDimensions(matrix1Columns, matrix2Columns);

            // Create a result matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> resultMatrix;
            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);

            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> BlasResultMatrix;
            BlasResultMatrix.setDimensions(matrix1Rows, matrix2Columns);

            // Fill the matrices
            //const double h_x = 1.0 / 100;
            //const double h_y = 1.0 / 100;

            for (int i = 0; i < matrix1Rows; i++) {
               for (int j = 0; j < matrix1Columns; j++) {
                  //double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  double value = 3*i;
                  denseMatrix1.setElement(i, j, value);
               }
            }

            for (int i = 0; i < matrix1Columns; i++) {
               for (int j = 0; j < matrix2Columns; j++) {
                  //double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  double value = 2*i;
                  denseMatrix2.setElement(i, j, value);
               }
            }

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "BLAS" },
                { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
                { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

#ifdef HAVE_BLAS
            auto matrixMultiplicationBenchmarkBLAS = [&]() mutable {
                matrixMultiplicationBLAS(denseMatrix1, denseMatrix2, BlasResultMatrix);
            };
            benchmark.time<Devices::Host>(device, matrixMultiplicationBenchmarkBLAS );
#endif //HAVE_BLAS

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "index type", TNL::getType<Index>() },
            { "device", device },
            { "algorithm", "TNL" },
            { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
            { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            auto matrixMultiplicationBenchmarkTNL = [&]() mutable {
               resultMatrix.getMatrixProduct(denseMatrix1, denseMatrix2, 1.0);
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType>> benchmarkMatricesCPU = { BlasResultMatrix };
            DenseMatricesResult<RealType, Devices::Host, IndexType> CPUResult(resultMatrix, benchmarkMatricesCPU);
            benchmark.time<Devices::Host>(device, matrixMultiplicationBenchmarkTNL,CPUResult);

            std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
      }

      std::cout << std::endl;
      std::cout << "=== Dense Matrix Trasnposition ===================================================================================================================" << std::endl;
      std::cout << std::endl;

      int dmatrix1Rows = 20; // Number of rows in matrix1 (same as columns in matrix2)
      int dmatrix1Columns = 1; // Number of columns in matrix1
      int numMatrices1 = 100;
      for (int i = 0; i < numMatrices1; ++i) {
         // Modify the matrix sizes for each iteration
         dmatrix1Rows += 10;
         dmatrix1Columns += 100;

         if (device == "cuda" || device == "all") {

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "MAGMA" },
               { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

#ifdef __CUDACC__
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> denseMatrix;
            denseMatrix.setDimensions(dmatrix1Rows, dmatrix1Columns);

            // Create a output matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> outputMatrix;
            outputMatrix.setDimensions(dmatrix1Columns, dmatrix1Rows);

            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> MagmaOutputMatrix;
            MagmaOutputMatrix.setDimensions(dmatrix1Columns, dmatrix1Rows);

            // Fill the matrix
            const double h_x = 1.0 / 100;
            const double h_y = 1.0 / 100;

            for (int i = 0; i < dmatrix1Rows; i++) {
               for (int j = 0; j < dmatrix1Columns; j++) {
                  double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  //double value = 3*i;
                  denseMatrix.setElement(i, j, value);
               }
            }

#ifdef HAVE_MAGMA
            // Lambda function to perform matrix transposition using MAGMA
            auto matrixTranspositionBenchmarkMagma = [&]() mutable {
               denseMatrixTransposeMAGMA(denseMatrix, MagmaOutputMatrix);
            };
            benchmark.time<Devices::Cuda>(device, matrixTranspositionBenchmarkMagma);
#endif //HAVE_MAGMA

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "TNL" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

            constexpr Index tileDim = 16; // Example tile dimension
            constexpr Index matrixProductCudaBlockSize = 256;
            constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = tileDim;
            launch_config.blockSize.y = cudaBlockRows;
            launch_config.dynamicSharedMemorySize = tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

            const Index rowTiles = roundUpDivision( dmatrix1Rows, tileDim );
            const Index columnTiles = roundUpDivision( dmatrix1Columns, tileDim );
            const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
            const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );


            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkTNL = [&]() mutable {
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
                                                                                 decltype(outputMatrixView),
                                                                                 decltype(denseMatrixView),
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
                                                                                    decltype(outputMatrixView),
                                                                                    decltype(denseMatrixView),
                                                                                    Real,
                                                                                    Index >;
                        Backend::launchKernelAsync(
                           kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y );
                     }
                  }
               Backend::streamSynchronize( launch_config.stream );
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesTransposition = {MagmaOutputMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> TranspositionResult(outputMatrix, benchmarkMatricesTransposition);
            benchmark.time<Devices::Cuda>(device, matrixTranspositionBenchmarkTNL,TranspositionResult);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "Combined (Test)" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkCombined = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++)
                  for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        // Determine if this particular segment of the matrix is aligned
                        bool isAligned = (gridIdx_x < columnGrids - 1 || denseMatrix.getColumns() % tileDim == 0)
                                       && (gridIdx_y < rowGrids - 1 || denseMatrix.getRows() % tileDim == 0);

                        auto outputMatrixView = outputMatrix.getView();
                        auto denseMatrixView = denseMatrix.getConstView();
                        constexpr auto kernel = DenseTranspositionKernel<tileDim,
                                                                        cudaBlockRows,
                                                                        decltype(outputMatrixView),
                                                                        decltype(denseMatrixView),
                                                                        Real,
                                                                        Index>;

                        // Launch the unified kernel with the isAligned parameter
                        Backend::launchKernelAsync(
                           kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y, isAligned);
                  }
               Backend::streamSynchronize(launch_config.stream);
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesTranspositionCombined = {MagmaOutputMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> TranspositionResult2(outputMatrix, benchmarkMatricesTranspositionCombined);
            benchmark.time<Devices::Cuda>(device, matrixTranspositionBenchmarkCombined,TranspositionResult2);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "Optimized" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkOptimized = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++)
                  for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Backend::getMaxGridXSize();
                        launch_config.gridSize.y = Backend::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

                        auto outputMatrixView = outputMatrix.getView();
                        auto denseMatrixView = denseMatrix.getConstView();
                        constexpr auto kernel = OptimizedDenseTranspositionKernel<tileDim,
                                                                                 cudaBlockRows,
                                                                                 decltype(outputMatrixView),
                                                                                 decltype(denseMatrixView),
                                                                                 Real,
                                                                                 Index>;

                        Backend::launchKernelAsync(
                           kernel, launch_config, outputMatrixView, denseMatrixView, 1.0, gridIdx_x, gridIdx_y);
                  }
               Backend::streamSynchronize(launch_config.stream);
            };
            std::vector<TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType>> benchmarkMatricesTranspositionOptimized = {MagmaOutputMatrix};
            DenseMatricesResult<RealType, Devices::Cuda, IndexType> TranspositionResult3(outputMatrix, benchmarkMatricesTranspositionOptimized);
            benchmark.time<Devices::Cuda>(device, matrixTranspositionBenchmarkOptimized,TranspositionResult3);

            std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
#endif //__CUDACC__
            }
            if (device == "host" || device == "all") {
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "TNL" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix;
            denseMatrix.setDimensions(dmatrix1Rows, dmatrix1Columns);

            // Create a output matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> outputMatrix;
            outputMatrix.setDimensions(dmatrix1Columns, dmatrix1Rows);

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkTNL = [&]() mutable {
               outputMatrix.getTransposition(denseMatrix);
            };
            benchmark.time<Devices::Host>(device, matrixTranspositionBenchmarkTNL);

            std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
         }
      return true;
      }
   };
}  // namespace TNL::Benchmarks::DenseMatrices
