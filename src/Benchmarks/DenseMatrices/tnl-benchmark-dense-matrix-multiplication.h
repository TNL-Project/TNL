// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Hip.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "DenseMatrixMultiplicationBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
}

template< typename Real >
bool
runDenseMatrixMultiplicationBenchmark( TNL::Config::ParameterContainer& parameters )
{
   TNL::Benchmarks::DenseMatrices::DenseMatrixMultiplicationBenchmark< Real > benchmark( parameters );
   benchmark.runBenchmark();
   return true;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );
   TNL::Devices::Host::configSetup( config );
#if defined( __CUDACC__ )
   TNL::Devices::Cuda::configSetup( config );
#elif defined( __HIP__ )
   TNL::Devices::Hip::configSetup( config );
#endif
   TNL::Benchmarks::DenseMatrices::DenseMatrixMultiplicationBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! TNL::Config::parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;
#if defined( __CUDACC__ )
   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;
#elif defined( __HIP__ )
   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Hip::setup( parameters ) )
      return EXIT_FAILURE;
#endif
   bool success = false;
   auto precision = parameters.getParameter< TNL::String >( "precision" );

   if( precision == "float" || precision == "all" ) {
      success = runDenseMatrixMultiplicationBenchmark< float >( parameters );
   }
   else if( precision == "double" || precision == "all" ) {
      success = runDenseMatrixMultiplicationBenchmark< double >( parameters );
   }
   else {
      std::cerr << "Unknown precision " << precision << ".\n";
      return EXIT_FAILURE;
   }

   return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
