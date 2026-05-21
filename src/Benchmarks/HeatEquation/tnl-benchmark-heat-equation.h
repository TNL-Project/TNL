// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>

#include "HeatEquationSolverBenchmarkParallelFor.h"
#include "HeatEquationSolverBenchmarkSimpleGrid.h"
#include "HeatEquationSolverBenchmarkGrid.h"
#include "HeatEquationSolverBenchmarkNdGrid.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );
   config.addDelimiter( "Heat equation benchmark settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the heat equation solver.", "grid" );
   config.addEntryEnum< TNL::String >( "parallel-for" );
   config.addEntryEnum< TNL::String >( "simple-grid" );
   config.addEntryEnum< TNL::String >( "grid" );
   config.addEntryEnum< TNL::String >( "nd-grid" );

   config.addDelimiter( "Device settings:" );
   config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
   config.addEntryEnum< TNL::String >( "host" );
   config.addEntryEnum< TNL::String >( "cuda" );
   config.addEntryEnum< TNL::String >( "hip" );
   config.addEntryEnum< TNL::String >( "sequential" );
   config.addEntryEnum< TNL::String >( "all" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );

   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   HeatEquationSolverBenchmark<>::configSetup( config );
}

template< typename Real, typename Device >
bool
startBenchmark( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   auto implementation = parameters.getParameter< TNL::String >( "implementation" );
   if( implementation == "parallel-for" ) {
      HeatEquationSolverBenchmarkParallelFor< Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   if( implementation == "simple-grid" ) {
      HeatEquationSolverBenchmarkSimpleGrid< Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   if( implementation == "grid" ) {
      HeatEquationSolverBenchmarkGrid< Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   if( implementation == "nd-grid" ) {
      HeatEquationSolverBenchmarkNdGrid< Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   return false;
}

template< typename Real >
bool
resolveDevice( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   auto device = parameters.getParameter< TNL::String >( "device" );
   bool result = true;
   if( device == "sequential" || device == "all" )
      result = startBenchmark< Real, TNL::Devices::Sequential >( benchmark, parameters ) && result;
   if( device == "host" || device == "all" )
      result = startBenchmark< Real, TNL::Devices::Host >( benchmark, parameters ) && result;
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      result = startBenchmark< Real, TNL::Devices::GPU >( benchmark, parameters ) && result;
#endif
   return result;
}

bool
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   auto precision = parameters.getParameter< TNL::String >( "precision" );
   bool result = true;
   if( precision == "all" || precision == "float" )
      result = resolveDevice< float >( benchmark, parameters ) && result;
   if( precision == "all" || precision == "double" )
      result = resolveDevice< double >( benchmark, parameters ) && result;
   return result;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
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
