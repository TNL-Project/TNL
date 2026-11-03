// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include "GraphBenchmarkSCC.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >(
      "precision", "Precision of the edge weights (ignored by SCC, which always uses double).", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
}

bool
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   // SCC ignores the edge weights, so the benchmark is instantiated only for one precision.
   // This halves the compilation time and memory of the GPU benchmark.
   const auto precision = parameters.getParameter< TNL::String >( "precision" );
   if( precision != "double" )
      std::cout << "SCC does not depend on the precision of the edge weights, running with double only.\n";
   TNL::Benchmarks::Graphs::GraphBenchmarkSCC< double > graphBenchmark( parameters );
   return graphBenchmark.runBenchmark( benchmark );
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   TNL::Benchmarks::Graphs::GraphBenchmarkSCC<>::configSetup( config );
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
