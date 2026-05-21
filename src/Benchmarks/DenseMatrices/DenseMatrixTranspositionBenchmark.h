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

#include "DenseMatricesResult.h"
#include "LegacyKernelsLauncher.h"
#include <cmath>
#include <utility>
#include <vector>

#if defined( __CUDACC__ )
   #include "MagmaBenchmark.h"
#endif

namespace TNL::Benchmarks::DenseMatrices {

template< typename Real = double, typename Index = int >
struct DenseMatrixTranspositionBenchmark
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
   DenseMatrixTranspositionBenchmark( TNL::Config::ParameterContainer parameters_ )
   : parameters( std::move( parameters_ ) )
   {}

   void
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark )
   {
      using HostMatrixType = TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType >;
      using GPUMatrixType = TNL::Matrices::DenseMatrix< RealType, Devices::GPU, IndexType >;

      const bool isLinearFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

      const auto device = parameters.getParameter< TNL::String >( "device" );

      IndexType dmatrix1Rows = 0;          // Number of rows in matrix1 (same as columns in matrix2)
      IndexType dmatrix1Columns = 0;       // Number of columns in matrix1
      const IndexType numMatrices1 = 100;  // Number of matrices that are going to be generated

      for( IndexType i = 0; i < numMatrices1; ++i ) {
         // Modify the matrix sizes for each iteration
         dmatrix1Rows += 100;
         dmatrix1Columns += 50;

#if defined( __CUDACC__ ) || defined( __HIP__ )
         if( device == "cuda" || device == "hip" || device == "all" ) {
            GPUMatrixType denseMatrix;
            denseMatrix.setDimensions( dmatrix1Rows, dmatrix1Columns );
            auto denseMatrixView = denseMatrix.getView();

            GPUMatrixType outputMatrix;
            outputMatrix.setDimensions( dmatrix1Columns, dmatrix1Rows );

            GPUMatrixType MagmaOutputMatrix;
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
            TNL::Algorithms::parallelFor< Devices::GPU >( 0, dmatrix1Columns, fill );
   #if defined( __CUDACC__ )
      #ifdef HAVE_MAGMA
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  {
                     { "index type", TNL::getType< Index >() },
                     { "real type", TNL::getType< Real >() },
                     { "device", device },
                     { "algorithm", "MAGMA" },
                     { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                  } ) );

            // Lambda function to perform matrix transposition using MAGMA
            auto matrixTranspositionBenchmarkMagma = [ & ]() mutable
            {
               denseMatrixTransposeMAGMA( denseMatrix, MagmaOutputMatrix );
            };
            benchmark.time< Devices::GPU >( device, matrixTranspositionBenchmarkMagma );
      #endif  //HAVE_MAGMA
   #endif

            bool LegacyOn = parameters.getParameter< TNL::String >( "include-legacy-kernels" ) == "legacy-on";
            if( LegacyOn ) {
               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     {
                        { "index type", TNL::getType< Index >() },
                        { "real type", TNL::getType< Real >() },
                        { "device", device },
                        { "algorithm", "Kernel 2.1" },
                        { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                     } ) );

               auto matrixTranspositionBenchmarkTNL = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixTranspositionKernel1( denseMatrix, outputMatrix );
               };
   #ifdef HAVE_MAGMA
               std::vector< GPUMatrixType > benchmarkMatricesTransposition = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > TranspositionResult(
                  outputMatrix, benchmarkMatricesTransposition );
   #endif  // HAVE_MAGMA
               benchmark.time< Devices::GPU >(
                  device,
                  matrixTranspositionBenchmarkTNL
   #ifdef HAVE_MAGMA
                  ,
                  TranspositionResult
   #endif  // HAVE_MAGMA
               );

               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     {
                        { "index type", TNL::getType< Index >() },
                        { "real type", TNL::getType< Real >() },
                        { "device", device },
                        { "algorithm", "Kernel 2.2" },
                        { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                     } ) );

               auto matrixTranspositionBenchmarkCombined = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, Devices::GPU, IndexType >::
                     launchMatrixTranspositionKernel2( denseMatrix, outputMatrix );
               };
   #ifdef HAVE_MAGMA
               std::vector< GPUMatrixType > benchmarkMatricesTranspositionCombined = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > TranspositionResult2(
                  outputMatrix, benchmarkMatricesTranspositionCombined );
   #endif  // HAVE_MAGMA
               benchmark.time< Devices::GPU >(
                  device,
                  matrixTranspositionBenchmarkCombined
   #ifdef HAVE_MAGMA
                  ,
                  TranspositionResult2
   #endif  // HAVE_MAGMA
               );

            }  // LegacyOn

            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  {
                     { "index type", TNL::getType< Index >() },
                     { "real type", TNL::getType< Real >() },
                     { "device", device },
                     { "algorithm", "Kernel 2.3" },
                     { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                  } ) );

            // Lambda function to perform matrix transposition using TNL
            auto matrixTranspositionBenchmarkFinal = [ & ]() mutable
            {
               outputMatrix.getTransposition( denseMatrix );
            };
   #ifdef HAVE_MAGMA
            std::vector< GPUMatrixType > benchmarkMatricesTranspositionFinal = { MagmaOutputMatrix };

            DenseMatricesResult< RealType, Devices::GPU, IndexType > TranspositionResult3(
               outputMatrix, benchmarkMatricesTranspositionFinal );
   #endif  // HAVE_MAGMA
            benchmark.time< Devices::GPU >(
               device,
               matrixTranspositionBenchmarkFinal
   #ifdef HAVE_MAGMA
               ,
               TranspositionResult3
   #endif  // HAVE_MAGMA
            );

            if( dmatrix1Rows == dmatrix1Columns ) {
               benchmark.setMetadataColumns(
                  TNL::Benchmarks::Benchmark::MetadataColumns(
                     {
                        { "index type", TNL::getType< Index >() },
                        { "real type", TNL::getType< Real >() },
                        { "device", device },
                        { "algorithm", "Kernel 2.4" },
                        { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                     } ) );

               // Lambda function to perform matrix transposition using TNL
               auto matrixTranspositionBenchmarkInPlace = [ & ]() mutable
               {
                  denseMatrix.getInPlaceTransposition();
               };
   #ifdef HAVE_MAGMA
               std::vector< GPUMatrixType > benchmarkMatricesTranspositionInPlace = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, Devices::GPU, IndexType > TranspositionResult4(
                  denseMatrix, benchmarkMatricesTranspositionInPlace );
   #endif  // HAVE_MAGMA
               benchmark.time< Devices::GPU >(
                  device,
                  matrixTranspositionBenchmarkInPlace
   #ifdef HAVE_MAGMA
                  ,
                  TranspositionResult4
   #endif  // HAVE_MAGMA
               );
            }
         }
#endif  // defined( __CUDACC__ ) || defined( __HIP__ )

         if( device == "host" || device == "all" ) {
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  {
                     { "index type", TNL::getType< Index >() },
                     { "real type", TNL::getType< Real >() },
                     { "device", device },
                     { "algorithm", "TNL" },
                     { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
                  } ) );

            HostMatrixType denseMatrix;
            denseMatrix.setDimensions( dmatrix1Rows, dmatrix1Columns );

            HostMatrixType outputMatrix;
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
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
         }
      }
   }
};

}  // namespace TNL::Benchmarks::DenseMatrices
