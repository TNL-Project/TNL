// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

#include "GridRegisterPressureBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Grid register pressure benchmark settings:" );
   config.addEntry< int >( "size", "Number of cells of the 3D grid along each axis.", 256 );
   config.addEntry< int >(
      "max-variables",
      "Maximal number of local variables in each cell (the benchmark runs 8, 16, 32, 48, 64, 96 and 128 variables).",
      128 );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addEntry< std::string >( "device", "Device the computation will run on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
}

template< typename Real >
void
resolveDevice( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   using namespace TNL::Benchmarks::GridRegisterPressure;
   runBenchmark< Real, TNL::Devices::Sequential >( benchmark, parameters );
   runBenchmark< Real, TNL::Devices::Host >( benchmark, parameters );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   runBenchmark< Real, TNL::Devices::GPU >( benchmark, parameters );
#endif
}

void
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );
   if( precision == "all" || precision == "float" )
      resolveDevice< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      resolveDevice< double >( benchmark, parameters );
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolvePrecision( benchmark, parameters );

   return EXIT_SUCCESS;
}
