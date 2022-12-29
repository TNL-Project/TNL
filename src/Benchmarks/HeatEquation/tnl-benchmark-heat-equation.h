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
#include "HeatEquationSolverBenchmarkNDArray.h"
#include "HeatEquationSolverBenchmarkGrid.h"

void configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the heat equation solver.", "grid" );
   config.addEntryEnum< TNL::String >( "parallel-for" );
   config.addEntryEnum< TNL::String >( "nd-array" );
   config.addEntryEnum< TNL::String >( "grid" );

   config.addDelimiter( "Device settings:" );
   config.addEntry<TNL::String>( "device", "Device the computation will run on.", "cuda" );
   config.addEntryEnum<TNL::String>( "all" );
   config.addEntryEnum<TNL::String>( "host" );
   config.addEntryEnum<TNL::String>( "sequential" );
   config.addEntryEnum<TNL::String>("cuda");
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );

   config.addDelimiter("Precision settings:");
   config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
   config.addEntryEnum("float");
   config.addEntryEnum("double");
   config.addEntryEnum("all");

   config.addDelimiter("Problem settings:");
   config.addEntry<int>("dimension", "Dimension of the benchmark problem.", 2);
}


template< int Dimension, typename Real, typename Device >
bool startBenchmark( TNL::Config::ParameterContainer& parameters )
{
   auto implementation = parameters.getParameter< TNL::String >( "implementation" );
   if( implementation == "parallel-for" )
   {
      HeatEquationSolverBenchmarkParallelFor< Dimension, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "nd-array" )
   {
      HeatEquationSolverBenchmarkNDArray< Dimension, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid" )
   {
      HeatEquationSolverBenchmarkGrid< Dimension, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   return false;
}

template< typename Real, typename Device >
bool resolveDimension( TNL::Config::ParameterContainer& parameters )
{
   const int dimension = parameters.getParameter< int >( "dimension" );
   if( dimension == 1 )
      return startBenchmark< 1, Real, Device >( parameters );
   if( dimension == 2 )
      return startBenchmark< 2, Real, Device >( parameters );
   if( dimension == 3 )
      return startBenchmark< 3, Real, Device >( parameters );
   std::cerr << "Wrong dimension " << dimension << " only 1D, 2D and 3D problems are allowed." << std::endl;
   return false;
}

template< typename Real >
bool resolveDevice( TNL::Config::ParameterContainer& parameters )
{
   auto device = parameters.getParameter<TNL::String>( "device" );
   if( device == "sequential" )
      return resolveDimension< Real, TNL::Devices::Sequential >( parameters );
   if( device == "host" )
      return resolveDimension< Real, TNL::Devices::Host >( parameters );
   if( device == "cuda" ) {
#ifdef HAVE_CUDA
      return resolveDimension< Real, TNL::Devices::Cuda >( parameters );
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
   HeatEquationSolverBenchmark<>::configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( !parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( !TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( !resolveReal( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
