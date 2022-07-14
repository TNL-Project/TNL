// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include "HeatEquationSolverBenchmarkParallelFor.h"
#include "HeatEquationSolverBenchmarkSimpleGrid.h"
#include "HeatEquationSolverBenchmarkGrid.h"
#include "HeatEquationSolverBenchmarkNdGrid.h"

void setupConfig( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the heat equation solver.", "grid" );
   config.addEntryEnum< TNL::String >( "parallel-for" );
   config.addEntryEnum< TNL::String >( "simple-grid" );
   config.addEntryEnum< TNL::String >( "grid" );
   config.addEntryEnum< TNL::String >( "nd-grid" );

   config.addDelimiter( "Device settings:" );
   config.addEntry<TNL::String>( "device", "Device the computation will run on.", "cuda" );
   config.addEntryEnum<TNL::String>( "all" );
   config.addEntryEnum<TNL::String>( "host" );
   config.addEntryEnum<TNL::String>( "sequential" );
#ifdef HAVE_CUDA
   config.addEntryEnum<TNL::String>("cuda");
#endif
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
   auto implementation = parameters.getParameter< TNL::String >( "implementation" );
   if( implementation == "parallel-for" )
   {
      HeatEquationSolverBenchmarkParallelFor< Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "simple-grid" )
   {
      HeatEquationSolverBenchmarkSimpleGrid< Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid" )
   {
      HeatEquationSolverBenchmarkGrid< Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "nd-grid" )
   {
      HeatEquationSolverBenchmarkNdGrid< Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   return false;
}

template< typename Real >
bool resolveDevice( TNL::Config::ParameterContainer& parameters )
{
   auto device = parameters.getParameter<TNL::String>( "device" );
   if( device == "host" )
      return startBenchmark< Real, TNL::Devices::Host >( parameters );
   if( device == "cuda" )
      return startBenchmark< Real, TNL::Devices::Cuda >( parameters );
   std::cerr << "Uknown device " << device << "." << std::endl;
   return false;
}

bool resolveReal( TNL::Config::ParameterContainer& parameters )
{
   auto precision = parameters.getParameter<TNL::String>( "precision" );
   if( precision == "float" )
      return resolveDevice< float >( parameters );
   if( precision == "double" )
      return resolveDevice< double >( parameters );
   std::cerr << "Uknonwn precison " << precision << "." << std::endl;
   return false;
}

int main(int argc, char* argv[])
{
   TNL::Config::ConfigDescription config;
   setupConfig( config );
   HeatEquationSolverBenchmark<>::setupConfig( config );

   TNL::Config::ParameterContainer parameters;

   if( !parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( !TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( !resolveReal( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
