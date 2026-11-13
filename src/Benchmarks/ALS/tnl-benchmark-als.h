// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmark.h>

#include "ALSBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "ALS benchmark settings:" );
   config.addRequiredEntry< std::string >(
      "input-file", "Confidences of the users (rows) and the items (columns) in the Matrix Market format." );
   config.addList< int >( "factors", "Numbers of factors.", { 32, 64, 128 } );
   config.addEntry< int >( "iterations", "Number of iterations of alternating least squares.", 10 );
   config.addEntry< double >( "regularization", "Regularization parameter lambda.", 0.01 );
   config.addEntry< int >( "cg-steps", "Number of steps of the conjugate gradient method.", 3 );
   config.addList< std::string >(
      "solver",
      "Computation of the products of the conjugate gradient method: the sampled and sparse-dense products (spmm), or "
      "both in one pass over the rows (fused).",
      { "all" } );
   config.addEntryEnum( "spmm" );
   config.addEntryEnum( "fused" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "float" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
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
run_benchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   TNL::Benchmarks::ALS::runBenchmark< Real, int >( benchmark, parameters );
}

void
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      run_benchmark< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      run_benchmark< double >( benchmark, parameters );
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

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolvePrecision( benchmark, parameters );

   return EXIT_SUCCESS;
}
