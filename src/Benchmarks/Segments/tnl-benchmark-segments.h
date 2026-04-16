// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "SegmentsBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   //config.addDelimiter( "General settings:" );

   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );
   TNL::Benchmarks::Segments::SegmentsBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   try {
      TNL::Benchmarks::Segments::SegmentsBenchmark< int > benchmark( parameters );
      benchmark.setupBenchmark();
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
