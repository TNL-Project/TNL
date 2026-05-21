// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

#include "BlasBenchmark.h"

#include "LegacyKernelsLauncher.h"
#include "DenseMatricesResult.h"
#include <cmath>
#include <utility>
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
struct DenseMatrixMultiplicationBenchmark
{
   using RealType = Real;
   using IndexType = Index;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Dense matrices benchmark settings:" );
      config.addEntry< TNL::String >( "fill-mode", "Method to fill matrices.", "linear" );
      config.addEntryEnum( "linear" );
      config.addEntryEnum( "trigonometric" );
      config.addEntry< TNL::String >( "include-legacy-kernels", "Include legacy kernels to the benchmark", "legacy-on" );
      config.addEntryEnum( "legacy-off" );
      config.addEntryEnum( "legacy-on" );
   }

   TNL::Config::ParameterContainer parameters;
   DenseMatrixMultiplicationBenchmark( TNL::Config::ParameterContainer parameters )
   : parameters( std::move( parameters ) )
   {}

   void
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark )
   {
      using HostMatrixType = TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType >;
      using GPUMatrixType = TNL::Matrices::DenseMatrix< RealType, Devices::GPU, IndexType >;

      const bool isLinearFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

      const auto device = parameters.getParameter< TNL::String >( "device" );

      const IndexType numMatrices = 50;  // Number of matrices for the cycle
      IndexType matrix1Rows = 0;         // Number of rows in matrix1
      IndexType matrix1Columns = 0;      // Number of columns in matrix1 && rows in matrix2
      IndexType matrix2Columns = 0;      // Number of columns in matrix2

      for( IndexType i = 0; i < numMatrices; ++i ) {
         // Modify the matrix sizes for each iteration
         matrix1Rows += 100;
         matrix1Columns += 100;
         matrix2Columns += 100;

#if defined( __CUDACC__ ) || defined( __HIP__ )
         if( device == "cuda" || device == "hip" || device == "all" ) {
            GPUMatrixType denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows, matrix1Columns );
            auto denseMatrix1View = denseMatrix1.getView();
            GPUMatrixType denseMatrix2;
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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix1Columns, fill1 );

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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix2Columns, fill2 );

            // Create result matrices
            GPUMatrixType resultMatrix;
            GPUMatrixType cuBLASResultMatrix;
            GPUMatrixType CutlassResultMatrix;
            GPUMatrixType MagmaResultMatrix;
            GPUMatrixType BlasResultMatrix;
            GPUMatrixType HipBlasResultMatrix;

            resultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            cuBLASResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            CutlassResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            MagmaResultMatrix.setDimensions( matrix1Rows, matrix2Columns );
            BlasResultMatrix.setDimensions( matrix1Rows, matrix2Columns );

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "cuBLAS" },
                    { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                    { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkcuBlas = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationCuBLAS( denseMatrix1, denseMatrix2, cuBLASResultMatrix, false, false );
            };
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkcuBlas );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            std::vector< GPUMatrixType > benchmarkMatricesMAGMA = { cuBLASResultMatrix };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > MagmaResult( MagmaResultMatrix, benchmarkMatricesMAGMA );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkMagma, MagmaResult );
      #endif  //HAVE_MAGMA

      #ifdef HAVE_CUTLASS
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            std::vector< GPUMatrixType > benchmarkMatricesCutlass = { cuBLASResultMatrix };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > CutlassResult(
               CutlassResultMatrix, benchmarkMatricesCutlass );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkCutlass, CutlassResult );
      #endif  //HAVE_CUTLASS
   #endif     // __CUDACC__

            bool LegacyOn = parameters.getParameter< TNL::String >( "include-legacy-kernels" ) == "legacy-on";
            if( LegacyOn ) {
               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.1" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               // Lambda function for the first kernel launch
               auto matrixMultiplicationBenchmarkOriginal = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel1( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesTNL = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                     ,
                                                                     MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                     ,
                                                                     CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > TNLResult( resultMatrix, benchmarkMatricesTNL );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkOriginal, TNLResult );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.2" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               // Lambda function for the optimized kernel launch
               auto matrixMultiplicationBenchmarkOptimized = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel2( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesTNL2 = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                      ,
                                                                      MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                      ,
                                                                      CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > TNL2Result( resultMatrix, benchmarkMatricesTNL2 );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkOptimized, TNL2Result );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.3" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               // Lambda function for the optimized kernel 2 launch
               auto matrixMultiplicationBenchmarkOptimized2 = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel3( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesSMA = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                     ,
                                                                     MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                     ,
                                                                     CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > SMAResult( resultMatrix, benchmarkMatricesSMA );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkOptimized2, SMAResult );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.4" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               // Lambda function for the optimized kernel launch
               auto matrixMultiplicationBenchmarkWarptiling = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel4( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesWarptiling = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                            ,
                                                                            MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                            ,
                                                                            CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > WarptilingResult(
                  resultMatrix, benchmarkMatricesWarptiling );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkWarptiling, WarptilingResult );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.5" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               // Lambda function for the optimized kernel launch
               auto matrixMultiplicationBenchmarkWarptiling2 = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel5( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesWarptiling2 = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                             ,
                                                                             MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                             ,
                                                                             CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > Warptiling2Result(
                  resultMatrix, benchmarkMatricesWarptiling2 );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkWarptiling2, Warptiling2Result );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "Kernel 1.6" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               auto matrixMultiplicationBenchmarkFermi = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel6( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesFermi = { cuBLASResultMatrix
   #ifdef HAVE_MAGMA
                                                                       ,
                                                                       MagmaResultMatrix
   #endif  // HAVE_MAGMA

   #ifdef HAVE_CUTLASS
                                                                       ,
                                                                       CutlassResultMatrix
   #endif  // HAVE_CUTLASS
               };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > FermiResult( resultMatrix, benchmarkMatricesFermi );
               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkFermi, FermiResult );

   #ifdef USE_TENSOR_CORES

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     { { "index type", TNL::getType< Index >() },
                       { "real type", TNL::getType< Real >() },
                       { "device", device },
                       { "algorithm", "TensorCores" },
                       { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                       { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

               resultMatrix.getValues() = 0;
               auto matrixMultiplicationBenchmarkTensorCores = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixMultiplicationKernel7( denseMatrix1, denseMatrix2, resultMatrix );
               };
               std::vector< GPUMatrixType > benchmarkMatricesTensorCores = { cuBLASResultMatrix
      #ifdef HAVE_MAGMA
                                                                             ,
                                                                             MagmaResultMatrix
      #endif  // HAVE_MAGMA

      #ifdef HAVE_CUTLASS
                                                                             ,
                                                                             CutlassResultMatrix
      #endif  // HAVE_CUTLASS
               };

               DenseMatricesResult< RealType, Devices::GPU, IndexType > TensorCoresResult(
                  resultMatrix, benchmarkMatricesTensorCores );

               benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkTensorCores, TensorCoresResult );
   #endif  // USE_TENSOR_CORES

            }  //LegacyOn

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "Final" },
                    { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                    { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            resultMatrix.getValues() = 0;
            auto matrixMultiplicationBenchmarkFinal = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1, denseMatrix2 );
            };
            std::vector< GPUMatrixType > benchmarkMatricesFinal = { cuBLASResultMatrix,
                                                                    MagmaResultMatrix,
                                                                    CutlassResultMatrix };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > FinalResult( resultMatrix, benchmarkMatricesFinal );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkFinal, FinalResult );

   #if defined( __HIP__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "HipBlas" },
                    { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                    { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkHIPBLAS = [ & ]() mutable
            {
               // Call cuBLAS matrix multiplication function
               matrixMultiplicationHIPBLAS( denseMatrix1, denseMatrix2, HipBlasResultMatrix, false, false );
            };
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkHIPBLAS );
   #endif
         }
#endif  // defined( __CUDACC__ ) || defined( __HIP__ )

         if( device == "host" || device == "all" ) {
            HostMatrixType denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows, matrix1Columns );

            HostMatrixType denseMatrix2;
            denseMatrix2.setDimensions( matrix1Columns, matrix2Columns );

            HostMatrixType resultMatrix;
            resultMatrix.setDimensions( matrix1Rows, matrix2Columns );

            HostMatrixType BlasResultMatrix;
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

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "TNL" },
                    { "matrix1 size", std::to_string( matrix1Rows ) + "x" + std::to_string( matrix1Columns ) },
                    { "matrix2 size", std::to_string( matrix1Columns ) + "x" + std::to_string( matrix2Columns ) } } ) );

            auto matrixMultiplicationBenchmarkTNL = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct( denseMatrix1, denseMatrix2, 1.0 );
            };
            std::vector< HostMatrixType > benchmarkMatricesCPU = { BlasResultMatrix };
            DenseMatricesResult< RealType, Devices::Host, IndexType > CPUResult( resultMatrix, benchmarkMatricesCPU );
            benchmark.time< Devices::Host >( device, matrixMultiplicationBenchmarkTNL, CPUResult );
         }
      }

#if defined( __CUDACC__ ) || defined( __HIP__ )
      const IndexType numMatrices2 = 100;  // Number of matrices for the cycle
      IndexType matrix1Rows2 = 1;          // Number of rows in matrix1
      IndexType matrix1Columns2 = 1;       // Number of columns in matrix1 && rows in matrix2
      IndexType matrix2Columns2 = 1;       // Number of columns in matrix2

      for( IndexType i = 0; i < numMatrices2; ++i ) {
         // Modify the matrix sizes for each iteration
         matrix1Rows2 += 2;
         matrix1Columns2 += 1;
         matrix2Columns2 += 3;

         // Multiplication with TransposeState
         if( device == "cuda" || device == "hip" || device == "all" ) {
            // Original Matrices
            GPUMatrixType denseMatrix1;
            denseMatrix1.setDimensions( matrix1Rows2, matrix1Columns2 );
            auto denseMatrix1View = denseMatrix1.getView();

            GPUMatrixType denseMatrix2;
            denseMatrix2.setDimensions( matrix1Columns2, matrix2Columns2 );  // Matches inner dimension of Matrix1
            auto denseMatrix2View = denseMatrix2.getView();

            // Transposed Matrix1 (For Transpose A Only and Transpose Both Matrices)
            GPUMatrixType denseMatrix1Transposed;
            denseMatrix1Transposed.setDimensions( matrix1Columns2, matrix1Rows2 );
            auto denseMatrix1TransposedView = denseMatrix1Transposed.getView();

            // Transposed Matrix2 (For Transpose B Only and Transpose Both Matrices)
            GPUMatrixType denseMatrix2Transposed;
            denseMatrix2Transposed.setDimensions(
               matrix2Columns2,
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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix1Columns2, fill1 );

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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix2Columns2, fill2 );

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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix1Rows2, fill1Transposed );

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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, matrix1Columns2, fill2Transposed );

            // Create result matrices
            GPUMatrixType resultMatrix;
            GPUMatrixType MagmaResultMatrix;
            GPUMatrixType CuBLASResultMatrix;
            GPUMatrixType HipBlasResultMatrix;

            resultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            MagmaResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            CuBLASResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );
            HipBlasResultMatrix.setDimensions( matrix1Rows2, matrix2Columns2 );

   #if defined( __HIP__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkHipBlasTransB );
   #endif

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkCuBlasTransA );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkMagmaTransA );
      #endif  //HAVE_MAGMA

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "tnlA" },
                    { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                    { "matrix2 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix2Columns2 ) } } ) );

            resultMatrix.getValues() = 0;
            auto matrixMultiplicationBenchmarkTransA = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct(
                  denseMatrix1Transposed,
                  denseMatrix2,
                  1.0,
                  TNL::Matrices::TransposeState::Transpose,
                  TNL::Matrices::TransposeState::None );
            };
            std::vector< GPUMatrixType > benchmarkMatricesATrans = { CuBLASResultMatrix
      #ifdef HAVE_MAGMA
                                                                     ,
                                                                     MagmaResultMatrix
      #endif  // HAVE_MAGMA
            };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > ATransResult( resultMatrix, benchmarkMatricesATrans );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkTransA, ATransResult );
   #endif

   #if defined( __HIP__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkHipBlasTransA );
   #endif

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkCuBlasTransB );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkMagmaTransB );
      #endif  //HAVE_MAGMA

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "tnlB" },
                    { "matrix1 size", std::to_string( matrix1Rows2 ) + "x" + std::to_string( matrix1Columns2 ) },
                    { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            resultMatrix.getValues() = 0;
            auto matrixMultiplicationBenchmarkTransB = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct(
                  denseMatrix1,
                  denseMatrix2Transposed,
                  1.0,
                  TNL::Matrices::TransposeState::None,
                  TNL::Matrices::TransposeState::Transpose );
            };
            std::vector< GPUMatrixType > benchmarkMatricesBTrans = { CuBLASResultMatrix
      #ifdef HAVE_MAGMA
                                                                     ,
                                                                     MagmaResultMatrix
      #endif  // HAVE_MAGMA
            };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > BTransResult( resultMatrix, benchmarkMatricesBTrans );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkTransB, BTransResult );
   #endif

   #if defined( __HIP__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkHipBlasTransBoth );
   #endif

   #if defined( __CUDACC__ )
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkCuBlasTransBoth );

      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
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
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkMagmaTransBoth );
      #endif  //HAVE_MAGMA

            resultMatrix.getValues() = 0;
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "index type", TNL::getType< Index >() },
                    { "real type", TNL::getType< Real >() },
                    { "device", device },
                    { "algorithm", "tnlAB" },
                    { "matrix1 size", std::to_string( matrix1Columns2 ) + "x" + std::to_string( matrix1Rows2 ) },
                    { "matrix2 size", std::to_string( matrix2Columns2 ) + "x" + std::to_string( matrix1Columns2 ) } } ) );

            auto matrixMultiplicationBenchmarkTransBoth = [ & ]() mutable
            {
               resultMatrix.getMatrixProduct(
                  denseMatrix1Transposed,
                  denseMatrix2Transposed,
                  1.0,
                  TNL::Matrices::TransposeState::Transpose,
                  TNL::Matrices::TransposeState::Transpose );
            };
            std::vector< GPUMatrixType > benchmarkMatricesBothTrans = { CuBLASResultMatrix
      #ifdef HAVE_MAGMA
                                                                        ,
                                                                        MagmaResultMatrix
      #endif  // HAVE_MAGMA
            };
            DenseMatricesResult< RealType, Devices::GPU, IndexType > BothTransResult(
               resultMatrix, benchmarkMatricesBothTrans );
            benchmark.time< Devices::GPU >( device, matrixMultiplicationBenchmarkTransBoth, BothTransResult );
   #endif
         }
      }
#endif  // defined( __CUDACC__ ) || defined( __HIP__ )
   }
};

}  // namespace TNL::Benchmarks::DenseMatrices
