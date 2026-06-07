// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

#include "DenseMatrixBenchmarkHelpers.h"
#include "DenseMatricesResult.h"
#include "LegacyKernelsLauncher.h"
#include "Wrappers/blasWrappers.h"

#include <vector>

#if defined( __CUDACC__ )
   #include "Wrappers/cublasWrappers.h"
   #include "Wrappers/cutlassWrappers.h"
   #include "Wrappers/magmaWrappers.h"
#elif defined( __HIP__ )
   #include "Wrappers/hipblasWrappers.h"
#endif

namespace TNL::Benchmarks::DenseMatrices {

template< typename Real, typename Index >
void
benchmarkGpuComputing(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >& matrix1,
   TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >& matrix2,
   TransposeState transposeA = TransposeState::None,
   TransposeState transposeB = TransposeState::None )
{
   using GPUMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >;
   using ReferenceVector = std::vector< typename DenseMatricesResult< Real, Devices::GPU, Index >::Reference >;

   const auto m1s = sizeString( matrix1, transposeA );
   const auto m2s = sizeString( matrix2, transposeB );
   const auto suffix = transposeSuffix( transposeA, transposeB );

   const auto cublasAlgo = "cuBLAS" + suffix;
   const auto magmaAlgo = "MAGMA" + suffix;
   const auto cutlassAlgo = "Cutlass" + suffix;
   const auto tnlAlgo = "TNL" + suffix;
   const auto hipblasAlgo = "hipBLAS" + suffix;

   const Index resultRows = transposeA == TransposeState::Transpose ? matrix1.getColumns() : matrix1.getRows();
   const Index resultCols = transposeB == TransposeState::Transpose ? matrix2.getRows() : matrix2.getColumns();

   GPUMatrixType resultMatrix;
   resultMatrix.setDimensions( resultRows, resultCols );

   GPUMatrixType cublasResultMatrix;
   cublasResultMatrix.setDimensions( resultRows, resultCols );

   GPUMatrixType magmaResultMatrix;
   magmaResultMatrix.setDimensions( resultRows, resultCols );

   GPUMatrixType cutlassResultMatrix;
   cutlassResultMatrix.setDimensions( resultRows, resultCols );

   GPUMatrixType hipblasResultMatrix;
   hipblasResultMatrix.setDimensions( resultRows, resultCols );

   auto buildReferencePairs = [ & ]() -> ReferenceVector
   {
      ReferenceVector pairs;
#if defined( __CUDACC__ )
      pairs.emplace_back( cublasResultMatrix, cublasAlgo );
#elif defined( __HIP__ )
      pairs.emplace_back( hipblasResultMatrix, hipblasAlgo );
#endif
#ifdef HAVE_MAGMA
      pairs.emplace_back( magmaResultMatrix, magmaAlgo );
#endif
#ifdef HAVE_CUTLASS
      pairs.emplace_back( cutlassResultMatrix, cutlassAlgo );
#endif
      return pairs;
   };

#if defined( __CUDACC__ )
   setMetadata< Real, Index >( benchmark, cublasAlgo, m1s, m2s );
   auto computeCuBLAS = [ & ]() mutable
   {
      matrixMultiplicationCuBLAS( matrix1, matrix2, cublasResultMatrix, transposeA, transposeB );
   };
   benchmark.time< Devices::GPU >( cublasAlgo, computeCuBLAS );

   #ifdef HAVE_MAGMA
   setMetadata< Real, Index >( benchmark, magmaAlgo, m1s, m2s );
   auto computeMAGMA = [ & ]() mutable
   {
      matrixMultiplicationMAGMA( matrix1, matrix2, magmaResultMatrix, transposeA, transposeB );
   };
   DenseMatricesResult< Real, Devices::GPU, Index > magmaResult( magmaResultMatrix, { { cublasResultMatrix, cublasAlgo } } );
   benchmark.time< Devices::GPU >( magmaAlgo, computeMAGMA, magmaResult );
   #endif

   #ifdef HAVE_CUTLASS
   if( suffix.empty() ) {
      setMetadata< Real, Index >( benchmark, cutlassAlgo, m1s, m2s );
      auto computeCutlass = [ & ]() mutable
      {
         matrixMultiplicationCutlass( matrix1, matrix2, cutlassResultMatrix );
      };
      DenseMatricesResult< Real, Devices::GPU, Index > cutlassResult(
         cutlassResultMatrix, { { cublasResultMatrix, cublasAlgo } } );
      benchmark.time< Devices::GPU >( cutlassAlgo, computeCutlass, cutlassResult );
   }
   #endif
#elif defined( __HIP__ )
   setMetadata< Real, Index >( benchmark, hipblasAlgo, m1s, m2s );
   auto computeHipBLAS = [ & ]() mutable
   {
      matrixMultiplicationHIPBLAS( matrix1, matrix2, hipblasResultMatrix, transposeA, transposeB );
   };
   benchmark.time< Devices::GPU >( hipblasAlgo, computeHipBLAS );
#endif

   if( suffix.empty() && parameters.getParameter< bool >( "include-legacy-kernels" ) ) {
      auto benchmarkLegacyKernel = [ & ]( const std::string& algoName, auto kernelLaunch )
      {
         setMetadata< Real, Index >( benchmark, algoName, m1s, m2s );
         resultMatrix.getValues() = 0;
         auto compute = [ & ]() mutable
         {
            kernelLaunch( matrix1, matrix2, resultMatrix );
         };
         DenseMatricesResult< Real, Devices::GPU, Index > legacyResult( resultMatrix, buildReferencePairs() );
         benchmark.time< Devices::GPU >( algoName, compute, legacyResult );
      };

      benchmarkLegacyKernel(
         "Kernel 1.1",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel1( m1, m2, r );
         } );
      benchmarkLegacyKernel(
         "Kernel 1.2",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel2( m1, m2, r );
         } );
      benchmarkLegacyKernel(
         "Kernel 1.3",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel3( m1, m2, r );
         } );
      benchmarkLegacyKernel(
         "Kernel 1.4",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel4( m1, m2, r );
         } );
      benchmarkLegacyKernel(
         "Kernel 1.5",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel5( m1, m2, r );
         } );
      benchmarkLegacyKernel(
         "Kernel 1.6",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel6( m1, m2, r );
         } );

#ifdef USE_TENSOR_CORES
      benchmarkLegacyKernel(
         "TensorCores",
         []( auto& m1, auto& m2, auto& r )
         {
            LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixMultiplicationKernel7( m1, m2, r );
         } );
#endif
   }

   setMetadata< Real, Index >( benchmark, tnlAlgo, m1s, m2s );
   resultMatrix.getValues() = 0;
   auto computeTnl = [ & ]() mutable
   {
      resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0, transposeA, transposeB );
   };
   DenseMatricesResult< Real, Devices::GPU, Index > tnlResult( resultMatrix, buildReferencePairs() );
   benchmark.time< Devices::GPU >( tnlAlgo, computeTnl, tnlResult );
}

template< typename Real, typename Index >
void
benchmarkHostStandard(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   Index m1Rows,
   Index m1Cols,
   Index m2Cols )
{
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >;

   const bool linFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";
   const auto m1s = std::to_string( m1Rows ) + "x" + std::to_string( m1Cols );
   const auto m2s = std::to_string( m1Cols ) + "x" + std::to_string( m2Cols );

   HostMatrixType denseMatrix1;
   denseMatrix1.setDimensions( m1Rows, m1Cols );
   fillHostMatrix< Real, Index >( denseMatrix1, linFill, 3 );

   HostMatrixType denseMatrix2;
   denseMatrix2.setDimensions( m1Cols, m2Cols );
   fillHostMatrix< Real, Index >( denseMatrix2, linFill, 3 );

   HostMatrixType resultMatrix;
   resultMatrix.setDimensions( m1Rows, m2Cols );

   HostMatrixType blasResultMatrix;
   blasResultMatrix.setDimensions( m1Rows, m2Cols );

#ifdef HAVE_BLAS
   setMetadata< Real, Index >( benchmark, "BLAS", m1s, m2s );
   auto computeBLAS = [ & ]() mutable
   {
      matrixMultiplicationBLAS( denseMatrix1, denseMatrix2, blasResultMatrix );
   };
   benchmark.time< Devices::Host >( "BLAS", computeBLAS );
#endif

   setMetadata< Real, Index >( benchmark, "TNL", m1s, m2s );
   auto computeTNL = [ & ]() mutable
   {
      resultMatrix.getMatrixProduct( denseMatrix1, denseMatrix2, 1.0 );
   };
#ifdef HAVE_BLAS
   DenseMatricesResult< Real, Devices::Host, Index > hostResult( resultMatrix, { { blasResultMatrix, "BLAS" } } );
   benchmark.time< Devices::Host >( "TNL", computeTNL, hostResult );
#else
   benchmark.time< Devices::Host >( "TNL", computeTNL );
#endif
}

template< typename Real, typename Index >
void
benchmarkGpuStandard(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   Index m1Rows,
   Index m1Cols,
   Index m2Cols )
{
   using GPUMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >;

   const bool linFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

   GPUMatrixType denseMatrix1;
   denseMatrix1.setDimensions( m1Rows, m1Cols );
   fillGpuMatrix( denseMatrix1, linFill, static_cast< Real >( 3 ) );

   GPUMatrixType denseMatrix2;
   denseMatrix2.setDimensions( m1Cols, m2Cols );
   fillGpuMatrix( denseMatrix2, linFill, static_cast< Real >( 2 ) );

   benchmarkGpuComputing< Real, Index >( benchmark, parameters, denseMatrix1, denseMatrix2 );
}

template< typename Real, typename Index >
void
benchmarkGpuTransposed(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   Index m1Rows,
   Index m1Cols,
   Index m2Cols )
{
   using GPUMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >;

   const bool linFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";

   GPUMatrixType denseMatrix1;
   denseMatrix1.setDimensions( m1Rows, m1Cols );
   fillGpuMatrix( denseMatrix1, linFill, static_cast< Real >( 3 ) );

   GPUMatrixType denseMatrix2;
   denseMatrix2.setDimensions( m1Cols, m2Cols );
   fillGpuMatrix( denseMatrix2, linFill, static_cast< Real >( 2 ) );

   GPUMatrixType denseMatrix1T;
   denseMatrix1T.setDimensions( m1Cols, m1Rows );
   fillGpuMatrix( denseMatrix1T, linFill, static_cast< Real >( 2 ) );

   GPUMatrixType denseMatrix2T;
   denseMatrix2T.setDimensions( m2Cols, m1Cols );
   fillGpuMatrix( denseMatrix2T, linFill, static_cast< Real >( 0 ) );

   benchmarkGpuComputing< Real, Index >(
      benchmark, parameters, denseMatrix1T, denseMatrix2, TransposeState::Transpose, TransposeState::None );
   benchmarkGpuComputing< Real, Index >(
      benchmark, parameters, denseMatrix1, denseMatrix2T, TransposeState::None, TransposeState::Transpose );
   benchmarkGpuComputing< Real, Index >(
      benchmark, parameters, denseMatrix1T, denseMatrix2T, TransposeState::Transpose, TransposeState::Transpose );
}

template< typename Real, typename Index >
void
runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto device = parameters.getParameter< TNL::String >( "device" );
   const Index minRows = parameters.getParameter< int >( "min-rows" );
   const Index maxRows = parameters.getParameter< int >( "max-rows" );
   const Index minCols = parameters.getParameter< int >( "min-columns" );
   const Index maxCols = parameters.getParameter< int >( "max-columns" );

   const Index numRowsSteps = 10;
   const Index numColSteps = 10;
   const Index rowStep = numRowsSteps > 0 ? ( maxRows - minRows ) / numRowsSteps : 0;
   const Index colStep = numColSteps > 0 ? ( maxCols - minCols ) / numColSteps : 0;

   for( Index r = 0; r <= numRowsSteps; ++r ) {
      const Index m1Rows = minRows + r * rowStep;
      for( Index c = 0; c <= numColSteps; ++c ) {
         const Index m1Cols = minCols + c * colStep;
         for( Index c2 = 0; c2 <= numColSteps; ++c2 ) {
            const Index m2Cols = minCols + c2 * colStep;

            if( device == "host" || device == "all" )
               benchmarkHostStandard< Real, Index >( benchmark, parameters, m1Rows, m1Cols, m2Cols );

#if defined( __CUDACC__ ) || defined( __HIP__ )
            if( device == "cuda" || device == "hip" || device == "all" ) {
               benchmarkGpuStandard< Real, Index >( benchmark, parameters, m1Rows, m1Cols, m2Cols );
               benchmarkGpuTransposed< Real, Index >( benchmark, parameters, m1Rows, m1Cols, m2Cols );
            }
#endif
         }
      }
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices
