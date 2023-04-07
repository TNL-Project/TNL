// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "GraphsBenchmark.h"

void configSetup( TNL::Config::ConfigDescription& config )
{
   //config.addDelimiter( "General settings:" );

   config.addDelimiter( "Device settings:" );
   config.addEntry< TNL::String >( "device", "Device the computation will run on.", "cuda" );
   config.addEntryEnum< TNL::String >( "all" );
   config.addEntryEnum< TNL::String >( "host" );
   config.addEntryEnum< TNL::String >( "sequential" );
   config.addEntryEnum< TNL::String >("cuda");
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );

   config.addDelimiter("Precision settings:");
   config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
   config.addEntryEnum("float");
   config.addEntryEnum("double");
   config.addEntryEnum("all");
}

template< typename Real, typename Device >
bool startBenchmark( TNL::Config::ParameterContainer& parameters )
{
   GraphsBenchmark< Real, Device > benchmark;
   return benchmark.runBenchmark( parameters );
}

template< typename Real >
bool resolveDevice( TNL::Config::ParameterContainer& parameters )
{
   auto device = parameters.getParameter<TNL::String>( "device" );
   if( device == "sequential" )
      return startBenchmark< Real, TNL::Devices::Sequential >( parameters );
   if( device == "host" )
      return startBenchmark< Real, TNL::Devices::Host >( parameters );
   if( device == "cuda" ) {
#ifdef __CUDACC__
      return startBenchmark< Real, TNL::Devices::Cuda >( parameters );
#else
      std::cerr << "The benchmark was not built with CUDA support." << std::endl;
      return false;
#endif
   }
   std::cerr << "Unknown device " << device << "." << std::endl;
   return false;
}

bool resolveReal( TNL::Config::ParameterContainer& parameters )
{
   auto precision = parameters.getParameter<TNL::String>( "precision" );
   if( precision == "float" )
      return resolveDevice< float >( parameters );
   if( precision == "double" )
      return resolveDevice< double >( parameters );
   std::cerr << "Unknown precision " << precision << "." << std::endl;
   return false;
}

int main(int argc, char* argv[])
{
   TNL::Config::ConfigDescription config;
   configSetup( config );
   GraphsBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( !parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( !TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( !resolveReal( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
