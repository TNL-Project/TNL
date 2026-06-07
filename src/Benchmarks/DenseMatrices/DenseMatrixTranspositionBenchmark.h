// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

#include "DenseMatrixBenchmarkHelpers.h"
#include "DenseMatricesResult.h"
#include "LegacyKernelsLauncher.h"

#include <vector>

#if defined( __CUDACC__ )
   #include "Wrappers/magmaWrappers.h"
#endif

namespace TNL::Benchmarks::DenseMatrices {

template< typename Real, typename Index >
void
benchmarkGpuTransposition(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   Index rows,
   Index cols )
{
   using GPUMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::GPU, Index >;

   const bool linFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";
   const auto sizeStr = std::to_string( rows ) + "x" + std::to_string( cols );

   GPUMatrixType denseMatrix;
   denseMatrix.setDimensions( rows, cols );
   fillGpuMatrix( denseMatrix, linFill, static_cast< Real >( 3 ) );

   GPUMatrixType outputMatrix;
   outputMatrix.setDimensions( cols, rows );

   GPUMatrixType magmaOutputMatrix;
   magmaOutputMatrix.setDimensions( cols, rows );

#if defined( __CUDACC__ )
   #ifdef HAVE_MAGMA
   setMetadata< Real, Index >( benchmark, "MAGMA", sizeStr );
   auto computeMagma = [ & ]() mutable
   {
      denseMatrixTransposeMAGMA( denseMatrix, magmaOutputMatrix );
   };
   benchmark.time< Devices::GPU >( "MAGMA", computeMagma );
   #endif
#endif

   if( parameters.getParameter< bool >( "include-legacy-kernels" ) ) {
      setMetadata< Real, Index >( benchmark, "Kernel 2.1", sizeStr );
      auto computeKernel1 = [ & ]() mutable
      {
         LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixTranspositionKernel1( denseMatrix, outputMatrix );
      };
#ifdef HAVE_MAGMA
      DenseMatricesResult< Real, Devices::GPU, Index > result1( outputMatrix, { { magmaOutputMatrix, "MAGMA" } } );
#endif
      benchmark.time< Devices::GPU >(
         "Kernel 2.1",
         computeKernel1
#ifdef HAVE_MAGMA
         ,
         result1
#endif
      );

      setMetadata< Real, Index >( benchmark, "Kernel 2.2", sizeStr );
      auto computeKernel2 = [ & ]() mutable
      {
         LegacyKernelsLauncher< Real, Devices::GPU, Index >::launchMatrixTranspositionKernel2( denseMatrix, outputMatrix );
      };
#ifdef HAVE_MAGMA
      DenseMatricesResult< Real, Devices::GPU, Index > result2( outputMatrix, { { magmaOutputMatrix, "MAGMA" } } );
#endif
      benchmark.time< Devices::GPU >(
         "Kernel 2.2",
         computeKernel2
#ifdef HAVE_MAGMA
         ,
         result2
#endif
      );
   }

   setMetadata< Real, Index >( benchmark, "Kernel 2.3", sizeStr );
   auto computeTnl = [ & ]() mutable
   {
      outputMatrix.getTransposition( denseMatrix );
   };
#ifdef HAVE_MAGMA
   DenseMatricesResult< Real, Devices::GPU, Index > tnlResult( outputMatrix, { { magmaOutputMatrix, "MAGMA" } } );
#endif
   benchmark.time< Devices::GPU >(
      "Kernel 2.3",
      computeTnl
#ifdef HAVE_MAGMA
      ,
      tnlResult
#endif
   );

   if( rows == cols ) {
      setMetadata< Real, Index >( benchmark, "Kernel 2.4", sizeStr );
      auto computeInPlace = [ & ]() mutable
      {
         denseMatrix.getInPlaceTransposition();
      };
#ifdef HAVE_MAGMA
      DenseMatricesResult< Real, Devices::GPU, Index > inPlaceResult( denseMatrix, { { magmaOutputMatrix, "MAGMA" } } );
#endif
      benchmark.time< Devices::GPU >(
         "Kernel 2.4",
         computeInPlace
#ifdef HAVE_MAGMA
         ,
         inPlaceResult
#endif
      );
   }
}

template< typename Real, typename Index >
void
benchmarkHostTransposition(
   TNL::Benchmarks::Benchmark& benchmark,
   const TNL::Config::ParameterContainer& parameters,
   Index rows,
   Index cols )
{
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >;

   const bool linFill = parameters.getParameter< TNL::String >( "fill-mode" ) == "linear";
   const auto sizeStr = std::to_string( rows ) + "x" + std::to_string( cols );

   HostMatrixType denseMatrix;
   denseMatrix.setDimensions( rows, cols );
   fillHostMatrix< Real, Index >( denseMatrix, linFill, 3 );

   HostMatrixType outputMatrix;
   outputMatrix.setDimensions( cols, rows );

   setMetadata< Real, Index >( benchmark, "TNL", sizeStr );
   auto computeTnl = [ & ]() mutable
   {
      outputMatrix.getTransposition( denseMatrix );
   };
   benchmark.time< Devices::Host >( "TNL", computeTnl );
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

   const Index numRowsSteps = 20;
   const Index numColSteps = 20;
   const Index rowStep = numRowsSteps > 0 ? ( maxRows - minRows ) / numRowsSteps : 0;
   const Index colStep = numColSteps > 0 ? ( maxCols - minCols ) / numColSteps : 0;

   for( Index r = 0; r <= numRowsSteps; ++r ) {
      const Index rows = minRows + r * rowStep;
      for( Index c = 0; c <= numColSteps; ++c ) {
         const Index cols = minCols + c * colStep;

#if defined( __CUDACC__ ) || defined( __HIP__ )
         if( device == "cuda" || device == "hip" || device == "all" )
            benchmarkGpuTransposition< Real, Index >( benchmark, parameters, rows, cols );
#endif

         if( device == "host" || device == "all" )
            benchmarkHostTransposition< Real, Index >( benchmark, parameters, rows, cols );
      }
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices
