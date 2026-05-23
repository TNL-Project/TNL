// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
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
startBenchmark( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   TNL::Benchmarks::Graphs::GraphBenchmarkBFS< Real > graphBenchmark( parameters );
   return graphBenchmark.runBenchmark( benchmark );
}

bool
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   auto precision = parameters.getParameter< TNL::String >( "precision" );
   bool result = true;
   if( precision == "all" || precision == "float" )
      result = startBenchmark< float >( benchmark, parameters ) && result;
   if( precision == "all" || precision == "double" )
      result = startBenchmark< double >( benchmark, parameters ) && result;
   return result;
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

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   if( ! resolvePrecision( benchmark, parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
