// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmark.h>

#include "SpMMBenchmark.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "SpMM benchmark settings:" );
   config.addEntry< std::string >( "input-file", "Input sparse matrix in the Matrix Market format.", "" );
   config.addEntry< std::string >(
      "matrix-generator", "Generated sparse matrix (used only when no input file is given).", "all" );
   config.addEntryEnum( "laplace-2d" );
   config.addEntryEnum( "laplace-3d" );
   config.addEntryEnum( "stencil-27" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "grid-size-2d", "Number of grid points along each axis of the generated 2D matrices.", 1024 );
   config.addEntry< int >( "grid-size-3d", "Number of grid points along each axis of the generated 3D matrices.", 100 );
   config.addList< int >(
      "dense-columns", "Numbers of columns of the dense matrix.", { 1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256 } );
   config.addEntry< std::string >( "dense-layout", "Layout of the dense matrices.", "all" );
   config.addEntryEnum( "row-major" );
   config.addEntryEnum( "column-major" );
   config.addEntryEnum( "all" );
   config.addList< std::string >(
      "kernels",
      "Benchmarked implementations: k-spmv, row-group, lanes-over-columns, nnz-split, binned, vendor (cuSPARSE or hipSPARSE) "
      "or all.",
      { "all" } );
   config.addEntry< std::string >(
      "row-order", "Order in which the RowGroup and LanesOverColumns kernels process the rows.", "all" );
   config.addEntryEnum( "natural" );
   config.addEntryEnum( "sorted" );
   config.addEntryEnum( "all" );
   config.addEntry< int >(
      "sorting-window", "Number of consecutive rows sorted by their lengths for the sorted row order (0 means all rows).", 0 );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
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
   TNL::Benchmarks::SpMM::runBenchmark< Real, int >( benchmark, parameters );
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
