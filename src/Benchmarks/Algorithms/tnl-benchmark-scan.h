// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>

#include "scan-common.h"

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Benchmarks::Scan;

void
resolveValueType( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& valueType = parameters.getParameter< std::string >( "value-type" );
   if( valueType == "all" || valueType == "char" )
      runScanBenchmark< char >( benchmark, parameters );
   if( valueType == "all" || valueType == "short" )
      runScanBenchmark< short >( benchmark, parameters );
   if( valueType == "all" || valueType == "int" )
      runScanBenchmark< int >( benchmark, parameters );
   if( valueType == "all" || valueType == "long" )
      runScanBenchmark< long >( benchmark, parameters );
   if( valueType == "all" || valueType == "float" )
      runScanBenchmark< float >( benchmark, parameters );
   if( valueType == "all" || valueType == "double" )
      runScanBenchmark< double >( benchmark, parameters );
   if( valueType == "all" || valueType == "double2" )
      runScanBenchmark< Containers::StaticVector< 2, double > >( benchmark, parameters );
}

// checks that all values of a list parameter are compiled into the benchmark
bool
checkCompiledValues( const Config::ParameterContainer& parameters, const std::string& name, const std::vector< int >& compiled )
{
   for( int value : parameters.getList< int >( name ) ) {
      if( std::find( compiled.begin(), compiled.end(), value ) == compiled.end() ) {
         std::cerr << "The value " << value << " of --" << name << " is not compiled into the benchmark. Compiled values:";
         for( int c : compiled )
            std::cerr << ' ' << c;
         std::cerr << '\n';
         return false;
      }
   }
   return true;
}

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );

   config.addDelimiter( "Scan benchmark settings:" );
   config.addEntry< std::string >( "device", "Device to run the benchmarks on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
#if defined( __CUDACC__ )
   config.addEntryEnum( "cuda" );
#elif defined( __HIP__ )
   config.addEntryEnum( "hip" );
#endif
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "value-type", "Value type of the scanned arrays.", "all" );
   config.addEntryEnum( "char" );
   config.addEntryEnum( "short" );
   config.addEntryEnum( "int" );
   config.addEntryEnum( "long" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "double2" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "scan-type", "Type of the scan.", "all" );
   config.addEntryEnum( "inclusive" );
   config.addEntryEnum( "exclusive" );
   config.addEntryEnum( "all" );
   config.addList< std::string >( "variants", "Variants of the scan to be measured.", { "all" } );
   config.addEntryEnum( "inplace" );
   config.addEntryEnum( "1-vector" );
   config.addEntryEnum( "2-vectors" );
   config.addEntryEnum( "3-vectors" );
   config.addEntryEnum( "all" );
   config.addList< std::string >( "libraries", "Libraries whose scans are measured (if they are available).", { "all" } );
   config.addEntryEnum( "tnl" );
   config.addEntryEnum( "std" );
   config.addEntryEnum( "thrust" );
   config.addEntryEnum( "cub" );
   config.addEntryEnum( "rocprim" );
   config.addEntryEnum( "all" );
   config.addList< std::string >(
      "gpu-kernels",
      "TNL scan kernels measured on the GPU: the public API with the default kernel selection ('default'), or the "
      "in-place scan with an explicitly selected kernel.",
      { "default" } );
   config.addEntryEnum( "default" );
   config.addEntryEnum( "lookback" );
   config.addEntryEnum( "scan-then-propagate" );
   config.addEntryEnum( "reduce-then-scan" );
   config.addEntryEnum( "all" );
   config.addList< int >(
      "values-per-thread", "Values per thread of the explicitly selected GPU kernels.", toVector( ValuesPerThreadList{} ) );
   config.addEntry< int >( "min-size", "Minimum size of the scanned arrays.", 100000 );
   config.addEntry< int >( "max-size", "Maximum size of the scanned arrays.", 25600000 );
   config.addEntry< int >(
      "size-step-factor",
      "Factor determining the sizes of the scanned arrays. First size is min-size and each following size is "
      "stepFactor*previousSize, up to max-size.",
      4 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::GPU::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( parameters.getParameter< int >( "size-step-factor" ) <= 1 ) {
      std::cerr << "The value of --size-step-factor must be greater than 1.\n";
      return EXIT_FAILURE;
   }
   if( ! checkCompiledValues( parameters, "values-per-thread", toVector( ValuesPerThreadList{} ) ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) || ! Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolveValueType( benchmark, parameters );

   return EXIT_SUCCESS;
}
