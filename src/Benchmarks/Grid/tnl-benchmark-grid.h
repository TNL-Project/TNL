
// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GridBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );
   config.addDelimiter( "Grid benchmark settings:" );
   for( int i = 0; i < 3; i++ )
      config.addEntry< int >( dimensionParameterIds[ i ], "Grid resolution.", 100 );

   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
   config.addEntryEnum< TNL::String >( "host" );
   config.addEntryEnum< TNL::String >( "sequential" );
   config.addEntryEnum< TNL::String >( "cuda" );
   config.addEntryEnum< TNL::String >( "hip" );
   config.addEntryEnum< TNL::String >( "all" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
}

template< typename Real, typename Device >
bool
startBenchmark( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
#ifndef GRID_DIM
   runBenchmark< 1, Real, Device >( benchmark, parameters );
   runBenchmark< 2, Real, Device >( benchmark, parameters );
   runBenchmark< 3, Real, Device >( benchmark, parameters );
#else
   runBenchmark< GRID_DIM, Real, Device >( benchmark, parameters );
#endif
   return true;
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
