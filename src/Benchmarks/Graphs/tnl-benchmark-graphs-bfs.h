// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "GraphBenchmarkBFS.h"

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
startBenchmark( TNL::Config::ParameterContainer& parameters, const std::string& programName )
{
   TNL::Benchmarks::Graphs::GraphBenchmarkBFS< Real > benchmark( parameters );
   return benchmark.runBenchmark( programName );
}

bool
resolveReal( TNL::Config::ParameterContainer& parameters, const std::string& programName )
{
   // BFS never uses edge weights, only connectivity, so it always benchmarks
   // a binary (bool) adjacency matrix regardless of --precision. The option
   // is still accepted (and ignored) so that run-tnl-benchmark-graphs, which
   // passes --precision unconditionally to every benchmark binary, keeps
   // working unmodified.
   return startBenchmark< bool >( parameters, programName );
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   TNL::Benchmarks::Graphs::GraphBenchmarkBFS<>::configSetup( config );
   configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( ! resolveReal( parameters, argv[ 0 ] ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
