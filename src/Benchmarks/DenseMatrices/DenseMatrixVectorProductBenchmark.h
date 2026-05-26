// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

#include "DenseMatrixBenchmarkHelpers.h"
#include "Wrappers/blasWrappers.h"

#if defined( __CUDACC__ )
   #include "Wrappers/cublasWrappers.h"
#elif defined( __HIP__ )
   #include "Wrappers/hipblasWrappers.h"
#endif

namespace TNL::Benchmarks::DenseMatrices {

template< typename Real, typename Index >
void
benchmarkHostVectorProduct(
   TNL::Benchmarks::Benchmark& benchmark,
   TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& matrix,
   TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& inVector,
   TNL::Containers::Vector< Real, TNL::Devices::Host, Index >& outVector )
{
   setMetadata< Real, Index >(
      benchmark, "TNL", std::to_string( matrix.getRows() ) + "x" + std::to_string( matrix.getColumns() ) );

   auto reset = [ & ]()
   {
      inVector = 1.0;
      outVector = 0.0;
   };

   auto compute = [ & ]()
   {
      matrix.vectorProduct( inVector, outVector );
   };
   benchmark.time< TNL::Devices::Host >( reset, "CPU", compute );

#ifdef HAVE_BLAS
   setMetadata< Real, Index >(
      benchmark, "BLAS", std::to_string( matrix.getRows() ) + "x" + std::to_string( matrix.getColumns() ) );

   auto computeBlas = [ & ]()
   {
      matrixVectorProductBLAS( matrix, inVector, outVector );
   };
   benchmark.time< TNL::Devices::Host >( reset, "CPU BLAS", computeBlas );
#endif
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Real, typename Index >
void
benchmarkGpuVectorProduct(
   TNL::Benchmarks::Benchmark& benchmark,
   const std::string& device,
   TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::ColumnMajorOrder >& matrixCMO,
   TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::RowMajorOrder >& matrixRMO,
   TNL::Containers::Vector< Real, TNL::Devices::GPU, Index >& inVector,
   TNL::Containers::Vector< Real, TNL::Devices::GPU, Index >& outVector1,
   TNL::Containers::Vector< Real, TNL::Devices::GPU, Index >& outVector2 )
{
   using GPUMatrixCMO =
      TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::ColumnMajorOrder >;
   using GPUMatrixRMO = TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::RowMajorOrder >;

   // Column-major
   fillGpuMatrix( matrixCMO, true, static_cast< Real >( 1.0 ) );
   setMetadata< Real, Index >(
      benchmark, "TNL CMO", std::to_string( matrixCMO.getRows() ) + "x" + std::to_string( matrixCMO.getColumns() ) );
   auto computeCMO = [ & ]()
   {
      matrixCMO.vectorProduct( inVector, outVector1 );
   };
   benchmark.time< TNL::Devices::GPU >( device, computeCMO );

   // Row-major
   fillGpuMatrix( matrixRMO, true, static_cast< Real >( 1.0 ) );
   setMetadata< Real, Index >(
      benchmark, "TNL RMO", std::to_string( matrixRMO.getRows() ) + "x" + std::to_string( matrixRMO.getColumns() ) );
   auto computeRMO = [ & ]()
   {
      matrixRMO.vectorProduct( inVector, outVector2 );
   };
   benchmark.time< TNL::Devices::GPU >( device, computeRMO );

   // cuBLAS / hipBLAS (column-major only)
   fillGpuMatrix( matrixCMO, true, static_cast< Real >( 1.0 ) );
   #if defined( __CUDACC__ )
   setMetadata< Real, Index >(
      benchmark, "cuBLAS", std::to_string( matrixCMO.getRows() ) + "x" + std::to_string( matrixCMO.getColumns() ) );
   auto computeCuBLAS = [ & ]()
   {
      matrixVectorProductCuBLAS( matrixCMO, inVector, outVector1 );
   };
   benchmark.time< TNL::Devices::GPU >( device, computeCuBLAS );
   #elif defined( __HIP__ )
   setMetadata< Real, Index >(
      benchmark, "hipBLAS", std::to_string( matrixCMO.getRows() ) + "x" + std::to_string( matrixCMO.getColumns() ) );
   auto computeHipBLAS = [ & ]()
   {
      matrixVectorProductHIPBLAS( matrixCMO, inVector, outVector1 );
   };
   benchmark.time< TNL::Devices::GPU >( device, computeHipBLAS );
   #endif
}

#endif

template< typename Real, typename Index >
void
runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );

   for( std::size_t rows = 10; rows <= 20000 * 20000; rows *= 2 ) {
      for( std::size_t columns = 10; columns <= 20000 * 20000; columns *= 2 ) {
         if( rows * columns > static_cast< std::size_t >( 20000 ) * 20000 )
            break;

         // don't abort due to out-of-memory errors
         try {
            if( device == "host" || device == "all" ) {
               using HostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >;
               using HostVector = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;

               HostMatrix hostMatrix;
               hostMatrix.setDimensions( rows, columns );
               fillHostMatrix< Real, Index >( hostMatrix, true, 1.0 );

               HostVector inHostVector( columns );
               HostVector outHostVector( rows );

               benchmark.setOperation( "GEMV", ( rows * columns + rows + columns ) * sizeof( Real ) / oneGB );
               benchmarkHostVectorProduct< Real, Index >( benchmark, hostMatrix, inHostVector, outHostVector );
            }

#if defined( __CUDACC__ ) || defined( __HIP__ )
            if( device == "cuda" || device == "hip" || device == "all" ) {
               using GPUMatrixCMO =
                  TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::ColumnMajorOrder >;
               using GPUMatrixRMO =
                  TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index, TNL::Algorithms::Segments::RowMajorOrder >;
               using GPUVector = TNL::Containers::Vector< Real, TNL::Devices::GPU, Index >;

               GPUMatrixCMO matrixCMO;
               GPUMatrixRMO matrixRMO;
               matrixCMO.setDimensions( rows, columns );
               matrixRMO.setDimensions( rows, columns );

               GPUVector inVector( columns );
               GPUVector outVector1( rows );
               GPUVector outVector2( rows );

   #if defined( __CUDACC__ )
               const std::string gpuDevice = ( device == "all" ) ? "cuda" : device;
   #elif defined( __HIP__ )
               const std::string gpuDevice = ( device == "all" ) ? "hip" : device;
   #endif
               benchmark.setOperation( "GEMV", ( rows * columns + rows + columns ) * sizeof( Real ) / oneGB );
               benchmarkGpuVectorProduct< Real, Index >(
                  benchmark, gpuDevice, matrixCMO, matrixRMO, inVector, outVector1, outVector2 );
            }
#endif
         }
         catch( TNL::Exceptions::BackendBadAlloc& ) {
         }
      }
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices
