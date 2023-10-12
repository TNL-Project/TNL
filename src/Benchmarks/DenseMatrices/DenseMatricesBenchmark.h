#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrix.hpp>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Matrices/MatrixOperations.h>

#include "CublasBenchmark.h" 
#include "BlasBenchmark.h"

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

        config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
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
            resultMatrix.setDimensions(matrix1Rows, matrix2Columns);
            auto matrixMultiplicationBenchmarkcuBlas = [&]() mutable {
                
                // Call cuBLAS matrix multiplication function
                matrixMultiplicationCuBLAS(denseMatrix1, denseMatrix2, resultMatrix);
                
            };
            benchmark.time<Devices::Cuda>(device, matrixMultiplicationBenchmarkcuBlas);

            
            
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

        }
     
         
        
        
    }
    return true;
    }
};

}  // namespace TNL::Benchmarks::DenseMatrices
