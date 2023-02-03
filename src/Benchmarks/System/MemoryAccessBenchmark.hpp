// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/parseCommandLine.h>
#include "TestArray.h"
#include "MemoryAccessBenchmark.h"

void
MemoryAccessBenchmark::
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter("Benchmark settings:");
   config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heat-equation.log");
   config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
   config.addEntryEnum("append");
   config.addEntryEnum("overwrite");
   config.addEntry<TNL::String>( "device", "TNL device type used for benchmarking.", "sequential" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntry< int >( "loops", "Number of iterations for every benchmark.", 10);
   config.addEntry< int >( "element-size", "Benchmark element size.", 1 );
   config.addEntry< int >( "min-array-size", "Minimal array size size.", 1 << 10 );
   config.addEntry< int >( "max-array-size", "Maximal array size size.", 1 << 30 );
   config.addEntry< int >( "threads-count", "Number of OpenMP threads for host device.", 1 );
   config.addEntry< TNL::String >( "access-type", "Type of memory accesses to be benchmarked.", "sequential" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "random" );
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
}

template< typename Device, int ElementSize >
bool
MemoryAccessBenchmark::
performBenchmark( const TNL::Config::ParameterContainer& parameters )
{
   auto output_mode = parameters.getParameter< TNL::String >( "output-mode" );
   auto log_file_name = parameters.getParameter< TNL::String >( "log-file" );
   auto loops = parameters.getParameter< int >( "loops" );
   auto verbose = parameters.getParameter< bool >( "verbose" );
   auto mode = std::ios::out;
   if( output_mode == "append" )
      mode |= std::ios::app;
   std::ofstream log_file( log_file_name.getString(), mode );
   TNL::Benchmarks::Benchmark<> benchmark(log_file, loops, verbose);

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, log_file_name, ".metadata.json" );

   auto device = std::is_same< Device, TNL::Devices::Sequential >::value ? "sequential" : "host";
   auto access_type = parameters.getParameter< TNL::String >( "access-type" );
   size_t min_size = parameters.getParameter< int >( "min-array-size" );
   size_t max_size = parameters.getParameter< int >( "max-array-size" );
   int threads_count = parameters.getParameter< int >( "threads-count" );
   const long long int elementsPerTest = max_size / sizeof( ElementSize );
   for( size_t size = min_size; size <= max_size; size *= 2 ) {
      benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
         { "element size", TNL::convertToString( ElementSize ) },
         { "array size", TNL::convertToString( size ) },
         { "access type", access_type }
      }));
      benchmark.setDatasetSize( size );
      TestArray< ElementSize > array( size );
      array.setThreadsCount( threads_count );
      array.setElementsPerTest( elementsPerTest );
      if( access_type == "sequential" )
         array.setupSequentialTest();
      else if( access_type == "random" )
         array.setupRandomTest();
      array.performTest();
      benchmark.setOperationsPerLoop( array.getTestedElementsCountPerThread() );

      auto compute = [&] () {
         array.performTest();
      };
      benchmark.time< Device >( device, compute );
   }
   return true;
}
