// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include "SegmentsBenchmark.h"

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   TNL::Benchmarks::Segments::SegmentsBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   try {
      // init benchmark
      TNL::Benchmarks::Benchmark benchmark;
      benchmark.setup( parameters, argv[ 0 ] );
      TNL::Benchmarks::Segments::SegmentsBenchmark< int > segmentsBenchmark( parameters );
      segmentsBenchmark.setupBenchmark( benchmark );
   }
   catch( std::exception& e ) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
   }
   catch( ... ) {
      std::cerr << "Unknown exception.\n";
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}
