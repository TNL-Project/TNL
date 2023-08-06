// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "GraphsBenchmark.h"

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

template< typename Real >
bool
startBenchmark( TNL::Config::ParameterContainer& parameters )
{
   TNL::Benchmarks::Graphs::GraphsBenchmark< Real > benchmark( parameters );
   return benchmark.runBenchmark();
}

bool
resolveReal( TNL::Config::ParameterContainer& parameters )
{
   auto precision = parameters.getParameter< TNL::String >( "precision" );
   if( precision == "float" )
      return startBenchmark< float >( parameters );
   if( precision == "double" )
      return startBenchmark< double >( parameters );
   std::cerr << "Unknown precision " << precision << "." << std::endl;
   return false;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );
   TNL::Benchmarks::Graphs::GraphsBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( ! resolveReal( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
