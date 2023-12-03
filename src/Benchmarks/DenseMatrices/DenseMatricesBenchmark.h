#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/detail/ParallelFor2D.h>

#include "CublasBenchmark.h"
#include "BlasBenchmark.h"
#include "CutlassBenchmark.h"
#include "MagmaBenchmark.h"
#include "DenseMatrixMultiplicationKernels.h"
#include <cmath>

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

      const int numMatrices = 100; //Number of matrices for the cycle
      int matrix1Rows = 20; // Number of rows in matrix1
      int matrix1Columns = 10; // Number of columns in matrix1 && rows in matrix 2
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

            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> denseMatrix1;
            denseMatrix1.setDimensions(matrix1Rows, matrix1Columns);

            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> denseMatrix2;
            denseMatrix2.setDimensions(matrix1Columns, matrix2Columns);

            // Fill the matrices
            const double h_x = 1.0 / 100;
            const double h_y = 1.0 / 100;

            for (int i = 0; i < matrix1Rows; i++) {
               for (int j = 0; j < matrix1Columns; j++) {
                  double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  denseMatrix1.setElement(i, j, value);
               }
            }

            for (int i = 0; i < matrix1Columns; i++) {
               for (int j = 0; j < matrix2Columns; j++) {
                  double value = std::sin(2 * M_PI * h_x * i) + std::cos(2 * M_PI * h_y * j);
                  denseMatrix2.setElement(i, j, value);
               }
            }

            // Create a result matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> resultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> cuBLASResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> CutlassResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> MagmaResultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> resultMatrix_mainTNL;
            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            cuBLASResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            CutlassResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            MagmaResultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            resultMatrix_mainTNL.setDimensions(matrix1Rows, matrix2Columns);

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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkMagma);
            /*
            if(TNL::l2Norm(MagmaResultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "MAGMA kernel does not match CuBLAS Kernel" << std::endl;
            }
            */
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkCutlass);
            /*
            if(TNL::l2Norm(CutlassResultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Cutlass kernel does not match CuBLAS Kernel" << std::endl;
            }
            */
#endif //HAVE_CUTLASS
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
               { "index type", TNL::getType<Index>() },
               { "device", device },
               { "algorithm", "TNL" },
               { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
               { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            constexpr Index tileDim = 16; // Example tile dimension, adjust as needed
            constexpr Index matrixProductCudaBlockSize = 256;
            constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
            Cuda::LaunchConfiguration launch_config;
            launch_config.blockSize.x = tileDim;
            launch_config.blockSize.y = cudaBlockRows;
            launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

            const Index rowTiles = roundUpDivision(matrix1Rows, tileDim);
            const Index columnTiles = roundUpDivision(matrix2Columns, tileDim);
            const Index rowGrids = roundUpDivision(rowTiles, Cuda::getMaxGridYSize());
            const Index columnGrids = roundUpDivision(columnTiles, Cuda::getMaxGridXSize());

            // Lambda function for the first kernel launch
            auto matrixMultiplicationBenchmarkOriginal = [&]() mutable {
               for (Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++) {
                     for (Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++) {
                        launch_config.gridSize.x = Cuda::getMaxGridXSize();
                        launch_config.gridSize.y = Cuda::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Cuda::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Cuda::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix_mainTNL.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Cuda::launchKernelAsync(DenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOriginal);
            /*
            if(TNL::l2Norm(resultMatrix_mainTNL.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL kernel does not match CuBLAS Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix_mainTNL.getValues() - MagmaResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL kernel does not match MAGMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix_mainTNL.getValues() - CutlassResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL kernel does not match Cutlass Kernel" << std::endl;
            }
            */
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
                        launch_config.gridSize.x = Cuda::getMaxGridXSize();
                        launch_config.gridSize.y = Cuda::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Cuda::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Cuda::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Cuda::launchKernelAsync(OptimizedDenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOptimized);
            /*
            if(TNL::l2Norm(resultMatrix.getValues() - resultMatrix_mainTNL.getValues()) > 1e-4)
            {
               std::cout << "Main kernel does not match TNL2 Kernel" << std::endl;
            }
            if(TNL::l2Norm(resultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL2 kernel does not match CuBLAS Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - MagmaResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL2 kernel does not match MAGMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - CutlassResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "TNL2 kernel does not match Cutlass Kernel" << std::endl;
            }
            */
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
                        launch_config.gridSize.x = Cuda::getMaxGridXSize();
                        launch_config.gridSize.y = Cuda::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Cuda::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Cuda::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();


                        Cuda::launchKernelAsync(Optimized2DenseMatrixProductKernel<tileDim, cudaBlockRows, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkOptimized2);
            /*
            if(TNL::l2Norm(resultMatrix.getValues() - resultMatrix_mainTNL.getValues()) > 1e-4)
            {
               std::cout << "Main kernel does not match 2D SMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "2D SMA kernel does not match CuBLAS Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - MagmaResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "2D SMA kernel does not match MAGMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - CutlassResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "2D SMA kernel does not match Cutlass Kernel" << std::endl;
            }
            */
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
                        launch_config.gridSize.x = Cuda::getMaxGridXSize();
                        launch_config.gridSize.y = Cuda::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Cuda::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Cuda::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Cuda::launchKernelAsync(WarpTilingDenseMatrixProductKernel<tileDim, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkWarptiling);
            /*
            if(TNL::l2Norm(resultMatrix.getValues() - resultMatrix_mainTNL.getValues()) > 1e-4)
            {
               std::cout << "Main kernel does not match Warptiling Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling kernel does not match CuBLAS Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - MagmaResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling kernel does not match MAGMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - CutlassResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling kernel does not match Cutlass Kernel" << std::endl;
            }
            */
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
                        launch_config.gridSize.x = Cuda::getMaxGridXSize();
                        launch_config.gridSize.y = Cuda::getMaxGridYSize();
                        if (gridIdx_x == columnGrids - 1)
                           launch_config.gridSize.x = columnTiles % Cuda::getMaxGridXSize();
                        if (gridIdx_y == rowGrids - 1)
                           launch_config.gridSize.y = rowTiles % Cuda::getMaxGridYSize();

                        auto resultMatrixView = resultMatrix.getView();
                        auto denseMatrix1View = denseMatrix1.getConstView();
                        auto denseMatrix2View = denseMatrix2.getConstView();

                        Cuda::launchKernelAsync(OptimizedWarpTilingDenseMatrixProductKernel<tileDim, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkWarptiling2);
            /*
            if(TNL::l2Norm(resultMatrix.getValues() - resultMatrix_mainTNL.getValues()) > 1e-4)
            {
               std::cout << "Main kernel does not match Warptiling2 Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - cuBLASResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling2 kernel does not match CuBLAS Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - MagmaResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling2 kernel does not match MAGMA Kernel" << std::endl;
            }

            if(TNL::l2Norm(resultMatrix.getValues() - CutlassResultMatrix.getValues()) > 1e-4)
            {
               std::cout << "Warptiling2 kernel does not match Cutlass Kernel" << std::endl;
            }
            */
            std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
#endif
         }

        if (device == "host" || device == "all") {
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "BLAS" },
                { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
                { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix1;
            denseMatrix1.setDimensions(matrix1Rows, matrix1Columns);

            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix2;
            denseMatrix2.setDimensions(matrix1Columns, matrix2Columns);

            // Create a result matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> resultMatrix;
            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);

#ifdef HAVE_BLAS
            auto matrixMultiplicationBenchmarkBLAS = [&]() mutable {
                matrixMultiplicationBLAS(denseMatrix1, denseMatrix2, resultMatrix);
            };
            benchmark.time<Devices::Host>(device, matrixMultiplicationBenchmarkBLAS );
#endif
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
            benchmark.time<Devices::Host>(device, matrixMultiplicationBenchmarkTNL);

            std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
       }

      std::cout << std::endl;
      std::cout << "=== Dense Matrix Trasnposition ===================================================================================================================" << std::endl;
      std::cout << std::endl;

      int dmatrix1Rows = 20; // Number of rows in matrix1 (same as columns in matrix2)
      int dmatrix1Columns = 10; // Number of columns in matrix1

       for (int i = 0; i < numMatrices; ++i) {
            // Modify the matrix sizes for each iteration
            dmatrix1Rows += 10;
            dmatrix1Columns += 10;

       if (device == "cuda" || device == "all") {
            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "MAGMA" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

#ifdef __CUDACC__
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix;
            denseMatrix.setDimensions(dmatrix1Rows, dmatrix1Columns);

            // Create a output matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> outputMatrix;
            outputMatrix.setDimensions(dmatrix1Columns, dmatrix1Rows);

#ifdef HAVE_MAGMA
            // Lambda function to perform matrix transposition using MAGMA
            auto matrixTranspositionBenchmarkMagma = [&]() mutable {
               denseMatrixTransposeMAGMA(denseMatrix, outputMatrix);
            };
            benchmark.time<Devices::Host>(device, matrixTranspositionBenchmarkMagma);

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
                { "index type", TNL::getType<Index>() },
                { "device", device },
                { "algorithm", "TNL" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));

#endif //HAVE_MAGMA

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkTNL = [&]() mutable {
               outputMatrix.getTransposition(denseMatrix);
            };
            benchmark.time<Devices::Host>(device, matrixTranspositionBenchmarkTNL);

            std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
#endif
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
