#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrix.hpp>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Matrices/MatrixOperations.h>

#include "CublasBenchmark.h"
#include "BlasBenchmark.h"
namespace TNL::Benchmarks::DenseMatrices {

//Main kernel for dense matrix multiplication
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

// Kernel 4: Optimized Vectorized Dense Matrix Product Kernel
template <int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2>
__global__ void VectorizedDenseMatrixProductKernel(ResultMatrix resultMatrix,
                                                            const Matrix1 matrixA,
                                                            const Matrix2 matrixB,
                                                            const typename ResultMatrix::RealType matrixMultiplicator) {
#ifdef __CUDACC__
    using IndexType = typename ResultMatrix::IndexType;
    using RealType = typename ResultMatrix::RealType;

    const IndexType matrixARows = matrixA.getRows();
    const IndexType matrixAColumns = matrixA.getColumns();
    const IndexType matrixBRows = matrixB.getRows();
    const IndexType matrixBColumns = matrixB.getColumns();
    const IndexType matrixCRows = resultMatrix.getRows();
    const IndexType matrixCColumns = resultMatrix.getColumns();

    // Shared memory with padding to avoid bank conflicts
    __shared__ RealType sharedTileA[tileDim][tileDim + 1];
    __shared__ RealType sharedTileB[tileDim][tileDim + 1];
    __shared__ RealType sharedTileC[tileDim][tileDim];

    const IndexType resultTileRow = blockIdx.y * tileDim;
    const IndexType resultTileColumn = blockIdx.x * tileDim;

    // Initialize shared memory tileC
    for (IndexType i = threadIdx.y; i < tileDim; i += blockDim.y) {
        for (IndexType j = threadIdx.x; j < tileDim; j += blockDim.x) {
            sharedTileC[i][j] = 0.0;
        }
    }
    __syncthreads();

    // Iterate over tiles with vectorized loads
    for (IndexType i = 0; i < matrixAColumns; i += tileDim) {
        // Vectorized load for matrix A
        if (resultTileRow + threadIdx.y < matrixARows && i + threadIdx.x < matrixAColumns) {
            const auto* tileAFloat4 = reinterpret_cast<const float4*>(&matrixA(resultTileRow + threadIdx.y, i));
            sharedTileA[threadIdx.y][threadIdx.x] = tileAFloat4->x;  // Assuming alignment is handled
        }

        // Vectorized load for matrix B
        if (i + threadIdx.y < matrixBRows && resultTileColumn + threadIdx.x < matrixBColumns) {
            const auto* tileBFloat4 = reinterpret_cast<const float4*>(&matrixB(i + threadIdx.y, resultTileColumn));
            sharedTileB[threadIdx.y][threadIdx.x] = tileBFloat4->x;  // Assuming alignment is handled
        }

        __syncthreads();

        // Matrix multiplication within the tile with loop unrolling
        #pragma unroll
        for (IndexType j = 0; j < tileDim; ++j) {
            #pragma unroll
            for (IndexType k = 0; k < tileDim; ++k) {
                sharedTileC[threadIdx.y][threadIdx.x] += matrixMultiplicator * sharedTileA[threadIdx.y][k] * sharedTileB[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write the result tile to the global memory
    for (IndexType i = threadIdx.y; i < tileDim; i += blockDim.y) {
        for (IndexType j = threadIdx.x; j < tileDim; j += blockDim.x) {
            const IndexType matrixCRow = resultTileRow + i;
            const IndexType matrixCColumn = resultTileColumn + j;
            if (matrixCRow < matrixCRows && matrixCColumn < matrixCColumns) {
                resultMatrix(matrixCRow, matrixCColumn) = sharedTileC[i][j];
            }
        }
    }
#endif
}

//kernel 5 (each warp (group of threads) is responsible for computing a subset of a tile)
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

//kernel 6 (padding in shared memory to reduce conflicts in shared memory)
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
        int matrix1Rows = 20; // Number of rows in matrix1 (same as columns in matrix2)
        int matrix1Columns = 10; // Number of columns in matrix1
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

            // Create a result matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> resultMatrix;
            TNL::Matrices::DenseMatrix<RealType, Devices::Cuda, IndexType> cuBLASResultMatrix;
            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            cuBLASResultMatrix.setDimensions(matrix1Rows, matrix2Columns);

            auto matrixMultiplicationBenchmarkcuBlas = [&]() mutable {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCuBLAS(denseMatrix1, denseMatrix2, cuBLASResultMatrix);
            };
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkcuBlas);

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

                        auto resultMatrixView = resultMatrix.getView();
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

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: TNL Kernel does not match CuBlas" << std::endl;
            }

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

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: TNL2 Kernel does not match CuBlas" << std::endl;
            }

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

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: 2 SMA Kernel does not match CuBlas" << std::endl;
            }

            benchmark.setMetadataColumns(TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "index type", TNL::getType<Index>() },
            { "device", device },
            { "algorithm", "Vectorized" },
            { "matrix1 size", std::to_string(matrix1Rows) + "x" + std::to_string(matrix1Columns)},
            { "matrix2 size", std::to_string(matrix1Columns) + "x" + std::to_string(matrix2Columns)}
            }));

            // Lambda function for the optimized kernel launch
            auto matrixMultiplicationBenchmarkVectorized = [&]() mutable {
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

                        Cuda::launchKernelAsync(VectorizedDenseMatrixProductKernel<tileDim, decltype(resultMatrixView), decltype(denseMatrix1View), decltype(denseMatrix2View)>,
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
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkVectorized);

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: Vectorized Kernel does not match CuBlas" << std::endl;
            }

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

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: Warptiling vector is NOT MIS" << std::endl;
            }

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

            if( cuBLASResultMatrix != resultMatrix ) {
               std::cout << "ERROR: Warptiling2 Kernel does not match CuBlas" << std::endl;
            }

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
                { "algorithm", "TNL" },
                { "matrix size", std::to_string(dmatrix1Rows) + "x" + std::to_string(dmatrix1Columns)},
            }));
#ifdef __CUDACC__
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> denseMatrix;
            denseMatrix.setDimensions(dmatrix1Rows, dmatrix1Columns);

            // Create a output matrix
            TNL::Matrices::DenseMatrix<RealType, Devices::Host, IndexType> outputMatrix;
            outputMatrix.setDimensions(dmatrix1Columns, dmatrix1Rows);

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
