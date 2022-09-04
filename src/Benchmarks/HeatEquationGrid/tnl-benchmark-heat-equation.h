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
#include "HeatEquationSolverBenchmarkParallelForShmem.h"
#include "HeatEquationSolverBenchmarkGridShmem.h"

void setupConfig( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the heat equation solver.", "grid" );
   config.addEntryEnum< TNL::String >( "parallel-for" );
   config.addEntryEnum< TNL::String >( "simple-grid" );
   config.addEntryEnum< TNL::String >( "grid" );
   config.addEntryEnum< TNL::String >( "nd-grid" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-16" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-32" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-64" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-128" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-256" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-512" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-1024" );
   config.addEntryEnum< TNL::String >( "parallel-for-shmem-2048" );
   config.addEntryEnum< TNL::String >( "grid-shmem-16" );
   config.addEntryEnum< TNL::String >( "grid-shmem-32" );
   config.addEntryEnum< TNL::String >( "grid-shmem-64" );
   config.addEntryEnum< TNL::String >( "grid-shmem-128" );
   config.addEntryEnum< TNL::String >( "grid-shmem-256" );
   config.addEntryEnum< TNL::String >( "grid-shmem-512" );
   config.addEntryEnum< TNL::String >( "grid-shmem-1024" );
   config.addEntryEnum< TNL::String >( "grid-shmem-2048" );

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
   if( implementation == "parallel-for-shmem-16" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 16, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-32" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 32, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-64" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 64, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-128" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 128, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   /*if( implementation == "parallel-for-shmem-256" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 256, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-512" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 512, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-1024" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 1024, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "parallel-for-shmem-2048" )
   {
      HeatEquationSolverBenchmarkParallelForShmem< 2048, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }*/

   if( implementation == "grid-shmem-16" )
   {
      HeatEquationSolverBenchmarkGridShmem< 16, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-32" )
   {
      HeatEquationSolverBenchmarkGridShmem< 32, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-64" )
   {
      HeatEquationSolverBenchmarkGridShmem< 64, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-128" )
   {
      HeatEquationSolverBenchmarkGridShmem< 128, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   /*if( implementation == "grid-shmem-256" )
   {
      HeatEquationSolverBenchmarkGridShmem< 256, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-512" )
   {
      HeatEquationSolverBenchmarkGridShmem< 512, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-1024" )
   {
      HeatEquationSolverBenchmarkGridShmem< 1024, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }
   if( implementation == "grid-shmem-2048" )
   {
      HeatEquationSolverBenchmarkGridShmem< 2048, Real, Device > benchmark;
      return benchmark.runBenchmark( parameters );
   }*/

   return false;
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
#ifdef HAVE_CUDA
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
   std::cerr << "Unknown precison " << precision << "." << std::endl;
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
