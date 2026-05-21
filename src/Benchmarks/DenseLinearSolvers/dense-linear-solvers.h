// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/fillRandom.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Solvers/Linear/GEM.h>
#include <TNL/Solvers/Linear/CuSolverWrapper.h>
#include <TNL/Math.h>

template< typename Real, typename Index >
void
benchmarkDenseLinearSolvers( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >;
   using HostVectorType = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;
   using HostMatrixPointer = std::shared_ptr< HostMatrixType >;

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
      TNL::Algorithms::fillRandom< TNL::Devices::Host >(
         input_matrix.getValues().getData(), input_matrix.getValues().getSize(), Real( -1 ), Real( 1 ) );
   }

   const auto matrixSize = input_matrix.getRows();
   HostVectorType host_b( matrixSize );
   HostVectorType host_x( matrixSize, 1 );
   input_matrix.vectorProduct( host_x, host_b );

   const auto& device = parameters.getParameter< std::string >( "device" );

   if( device == "host" || device == "all" ) {
      // Benchmark GEM on CPU
      HostMatrixPointer host_matrix = std::make_shared< HostMatrixType >( input_matrix );
      TNL::Solvers::Linear::GEM< HostMatrixType > host_gem;
      host_gem.setMatrix( host_matrix );
      host_gem.setPivoting( parameters.getParameter< bool >( "pivoting" ) );
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
   }

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" ) {
      using GPUMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::GPU, Index >;
      using GPUVectorType = TNL::Containers::Vector< Real, TNL::Devices::GPU, Index >;
      using GPUMatrixPointer = std::shared_ptr< GPUMatrixType >;

      GPUMatrixPointer gpu_matrix = std::make_shared< GPUMatrixType >();
      *gpu_matrix = input_matrix;
      GPUVectorType gpu_b( host_b );
      GPUVectorType gpu_x( matrixSize, 0 );

      TNL::Solvers::Linear::GEM< GPUMatrixType > gpu_gem;
      gpu_gem.setMatrix( gpu_matrix );
      gpu_gem.setPivoting( parameters.getParameter< bool >( "pivoting" ) );
      gpu_gem.setSolverMonitor( benchmark.getMonitor() );

      // reset function
      auto reset_gpu = [ & ]()
      {
         gpu_x = 0;
         *gpu_matrix = input_matrix;  // Reset the matrix to the original state
      };

      // benchmark function
      auto compute_gpu = [ & ]()
      {
         const bool converged = gpu_gem.solve( gpu_b, gpu_x );
         if( ! converged )
            throw std::runtime_error( "GPU solver did not converge" );
      };
      benchmark.time< TNL::Devices::GPU >( reset_gpu, "GPU", compute_gpu );

      // Benchmark CuSolverWrapper on GPU
      TNL::Solvers::Linear::CuSolverWrapper< GPUMatrixType > cuSolverWrapper;
      cuSolverWrapper.setMatrix( gpu_matrix );
      cuSolverWrapper.setSolverMonitor( benchmark.getMonitor() );

      // benchmark function
      auto compute_cuSolverWrapper = [ & ]()
      {
         const bool converged = cuSolverWrapper.solve( gpu_b, gpu_x );
         if( ! converged )
            throw std::runtime_error( "CuSolverWrapper solver did not converge" );
      };
      benchmark.time< TNL::Devices::GPU >( reset_gpu, "CuSolverWrapper", compute_cuSolverWrapper );
   }
#endif
}
