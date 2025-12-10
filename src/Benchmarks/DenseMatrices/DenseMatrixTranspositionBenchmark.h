// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

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
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-dense-matrix-transposition.log" );
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
      config.addEntry< TNL::String >( "include-legacy-kernels", "Include legacy kernels to the benchmark", "legacy-on" );
      config.addEntryEnum( "legacy-off" );
      config.addEntryEnum( "legacy-on" );
   }

   TNL::Config::ParameterContainer parameters;
   DenseMatrixTranspositionBenchmark( TNL::Config::ParameterContainer parameters_ )
   : parameters( std::move( parameters_ ) )
   {}

   bool
   runBenchmark()
   {
      const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const auto outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const IndexType loops = parameters.getParameter< IndexType >( "loops" );
      const IndexType verbose = parameters.getParameter< IndexType >( "verbose" );
      const bool isLinearFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      const auto device = parameters.getParameter< TNL::String >( "device" );

      IndexType dmatrix1Rows = 0;     // Number of rows in matrix1 (same as columns in matrix2)
      IndexType dmatrix1Columns = 0;  // Number of columns in matrix1
      IndexType numMatrices1 = 100;   // NUmber of matrices that are going to be generated
      for( IndexType i = 0; i < numMatrices1; ++i ) {
         // Modify the matrix sizes for each iteration
         dmatrix1Rows += 100;
         dmatrix1Columns += 50;

         if( device == "cuda" || device == "hip" || device == "all" ) {
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
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
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
            benchmark.time< DeviceType >( device, matrixTranspositionBenchmarkMagma );
      #endif  //HAVE_MAGMA
   #endif

            bool LegacyOn = parameters.getParameter< TNL::String >( "include-legacy-kernels" ) == "legacy-on";
            if( LegacyOn ) {
               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                  { "index type", TNL::getType< Index >() },
                  { "real type", TNL::getType< Real >() },
                  { "device", device },
                  { "algorithm", "Kernel 2.1" },
                  { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
               } ) );

               auto matrixTranspositionBenchmarkTNL = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, DeviceType, IndexType >::
                     launchMatrixTranspositionKernel1( denseMatrix, outputMatrix );
               };
   #ifdef HAVE_MAGMA
               std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTransposition = {
                  MagmaOutputMatrix
               };
               DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult( outputMatrix,
                                                                                           benchmarkMatricesTransposition );
   #endif  // HAVE_MAGMA
               benchmark.time< DeviceType >( device,
                                             matrixTranspositionBenchmarkTNL
   #ifdef HAVE_MAGMA
                                             ,
                                             TranspositionResult
   #endif  // HAVE_MAGMA
               );

               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                  { "index type", TNL::getType< Index >() },
                  { "real type", TNL::getType< Real >() },
                  { "device", device },
                  { "algorithm", "Kernel 2.2" },
                  { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
               } ) );

               auto matrixTranspositionBenchmarkCombined = [ & ]() mutable
               {
                  TNL::Benchmarks::DenseMatrices::LegacyKernelsLauncher< RealType, DeviceType, IndexType >::
                     launchMatrixTranspositionKernel2( denseMatrix, outputMatrix );
               };
   #ifdef HAVE_MAGMA
               std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > >
                  benchmarkMatricesTranspositionCombined = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult2(
                  outputMatrix, benchmarkMatricesTranspositionCombined );
   #endif  // HAVE_MAGMA
               benchmark.time< DeviceType >( device,
                                             matrixTranspositionBenchmarkCombined
   #ifdef HAVE_MAGMA
                                             ,
                                             TranspositionResult2
   #endif  // HAVE_MAGMA
               );

            }  // LegacyOn

            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
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
            std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > > benchmarkMatricesTranspositionFinal = {
               MagmaOutputMatrix
            };

            DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult3( outputMatrix,
                                                                                         benchmarkMatricesTranspositionFinal );
   #endif  // HAVE_MAGMA
            benchmark.time< DeviceType >( device,
                                          matrixTranspositionBenchmarkFinal
   #ifdef HAVE_MAGMA
                                          ,
                                          TranspositionResult3
   #endif  // HAVE_MAGMA
            );

            if( dmatrix1Rows == dmatrix1Columns ) {
               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
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
               std::vector< TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > >
                  benchmarkMatricesTranspositionInPlace = { MagmaOutputMatrix };
               DenseMatricesResult< RealType, DeviceType, IndexType > TranspositionResult4(
                  denseMatrix, benchmarkMatricesTranspositionInPlace );
   #endif  // HAVE_MAGMA
               benchmark.time< DeviceType >( device,
                                             matrixTranspositionBenchmarkInPlace
   #ifdef HAVE_MAGMA
                                             ,
                                             TranspositionResult4
   #endif  // HAVE_MAGMA
               );
            }

#endif  // ( __CUDACC__ ) || defined( __HIP__ )
         }
         if( device == "host" || device == "all" ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "index type", TNL::getType< Index >() },
               { "real type", TNL::getType< Real >() },
               { "device", device },
               { "algorithm", "TNL" },
               { "matrix size", std::to_string( dmatrix1Rows ) + "x" + std::to_string( dmatrix1Columns ) },
            } ) );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > denseMatrix;
            denseMatrix.setDimensions( dmatrix1Rows, dmatrix1Columns );

            TNL::Matrices::DenseMatrix< RealType, Devices::Host, IndexType > outputMatrix;
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
      return true;
   }
};
}  // namespace TNL::Benchmarks::DenseMatrices
