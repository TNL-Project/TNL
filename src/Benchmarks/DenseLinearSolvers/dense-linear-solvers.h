// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <thread>

#include <TNL/Algorithms/fillRandom.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Solvers/Linear/GEM.h>
#include <TNL/Solvers/Linear/CuSolverWrapper.h>
#include <TNL/Math.h>

template< typename Real, typename Index >
void
benchmarkDenseLinearSolvers( TNL::Config::ParameterContainer& parameters )
{
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >;
   using HostVectorType = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;
   using HostMatrixPointer = std::shared_ptr< HostMatrixType >;

   const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   auto mode = std::ios::out;
   if( parameters.getParameter< bool >( "append-log" ) )
      mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );
   TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );
   benchmark.getMonitor().setRefreshRate( 1000 );  // refresh rate in milliseconds
   benchmark.getMonitor().setStage( "GEM elimination stage:" );

   HostMatrixType input_matrix;
   if( parameters.checkParameter( "input-matrix" ) ) {
      TNL::Matrices::MatrixReader< HostMatrixType > reader;
      reader.readMtx( parameters.getParameter< TNL::String >( "input-file" ), input_matrix );
   }
   else {
      auto matrixSize = parameters.getParameter< int >( "matrix-size" );
      input_matrix.setDimensions( matrixSize, matrixSize );
      if( verbose > 1 )
         std::cout << "Creating random matrix of size " << matrixSize << "x" << matrixSize << '\n';
      TNL::Algorithms::fillRandom< TNL::Devices::Host >(
         input_matrix.getValues().getData(), input_matrix.getValues().getSize(), Real( -1 ), Real( 1 ) );
   }

   const auto matrixSize = input_matrix.getRows();
   HostVectorType host_b( matrixSize );
   HostVectorType host_x( matrixSize, 1 );
   input_matrix.vectorProduct( host_x, host_b );

   // Benchmark GEM on CPU
   HostMatrixPointer host_matrix = std::make_shared< HostMatrixType >( input_matrix );
   TNL::Solvers::Linear::GEM< HostMatrixType > host_gem;
   host_gem.setMatrix( host_matrix );
   host_gem.setPivoting( parameters.getParameter< TNL::String >( "pivoting" ) == "yes" );
   host_gem.setSolverMonitor( benchmark.getMonitor() );

   // reset function
   auto reset_host = [ & ]()
   {
      host_x = 0;
      *host_matrix = input_matrix;  // Reset the matrix to the original state
   };

   // benchmark function
   auto compute_host = [ & ]()
   {
      const bool converged = host_gem.solve( host_b, host_x );
      if( ! converged )
         throw std::runtime_error( "CPU solver did not converge" );
   };
   benchmark.time< TNL::Devices::Host >( reset_host, "CPU", compute_host );
   if( max( host_x - 1 ) > 1e-5 )
      std::cout << "Warning: the result of the CPU solver is not equal to the expected result: " << max( host_x - 1 )
                << '\n';

#ifdef __CUDACC__
   using CudaMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda, Index >;
   using CudaVectorType = TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >;
   using CudaMatrixPointer = std::shared_ptr< CudaMatrixType >;

   CudaMatrixPointer cuda_matrix = std::make_shared< CudaMatrixType >();
   *cuda_matrix = input_matrix;
   CudaVectorType cuda_b( host_b ), cuda_x( matrixSize, 0 );

   TNL::Solvers::Linear::GEM< CudaMatrixType > cuda_gem;
   cuda_gem.setMatrix( cuda_matrix );
   cuda_gem.setPivoting( parameters.getParameter< TNL::String >( "pivoting" ) == "yes" );
   cuda_gem.setSolverMonitor( benchmark.getMonitor() );

   // reset function
   auto reset_cuda = [ & ]()
   {
      cuda_x = 0;
      *cuda_matrix = input_matrix;  // Reset the matrix to the original state
   };

   // benchmark function
   auto compute_cuda = [ & ]()
   {
      const bool converged = cuda_gem.solve( cuda_b, cuda_x );
      if( ! converged )
         throw std::runtime_error( "CUDA solver did not converge" );
   };
   benchmark.time< TNL::Devices::Cuda >( reset_cuda, "GPU", compute_cuda );

   // Benchmark CuSolverWrapper on GPU
   TNL::Solvers::Linear::CuSolverWrapper< CudaMatrixType > cuSolverWrapper;
   cuSolverWrapper.setMatrix( cuda_matrix );
   cuSolverWrapper.setSolverMonitor( benchmark.getMonitor() );
   // reset function
   auto reset_cuSolverWrapper = [ & ]()
   {
      cuda_x = 0;
      *cuda_matrix = input_matrix;  // Reset the matrix to the original state
   };

   // benchmark function
   auto compute_cuSolverWrapper = [ & ]()
   {
      const bool converged = cuSolverWrapper.solve( cuda_b, cuda_x );
      if( ! converged )
         throw std::runtime_error( "CuSolverWrapper solver did not converge" );
   };
   benchmark.time< TNL::Devices::Cuda >( reset_cuSolverWrapper, "CuSolverWrapper", compute_cuSolverWrapper );
#endif
}
