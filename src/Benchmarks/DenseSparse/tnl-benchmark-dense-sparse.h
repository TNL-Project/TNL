// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmark.h>

#include "DenseSparseBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Dense-sparse product benchmark settings:" );
   config.addRequiredEntry< std::string >( "input-file", "The sparse matrix A in the Matrix Market format." );
   config.addList< int >( "dense-rows", "Numbers of rows of the dense matrix B.", { 1, 2, 4, 8, 16, 32, 64, 128 } );
   config.addList< std::string >(
      "method", "Methods of the product: transpose-all, transpose-dense, transposed-A, atomic or all.", { "all" } );
   config.addEntry< std::string >( "organization", "Organization of the dense matrices B and C.", "all" );
   config.addEntryEnum( "row-major" );
   config.addEntryEnum( "column-major" );
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
   TNL::Benchmarks::DenseSparse::runBenchmark< Real, int >( benchmark, parameters );
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
