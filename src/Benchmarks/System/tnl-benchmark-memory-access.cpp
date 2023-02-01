// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/parseCommandLine.h>
#include "TestArray.h"

void configSetup( TNL::Config::ConfigDescription& config )
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
   config.addEntry< int >( "min-array-size", "Minimal array size size. Zero means that minimal array size is set to the cache line size.", 0 );
   config.addEntry< int >( "max-array-size", "Maximal array size size.", 1 << 30 );
   config.addEntry< TNL::String >( "access-type", "Type of memory accesses to be benchmarked.", "sequential" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "random" );
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
}

template< typename Device, int ElementSize >
bool performBenchmark( const TNL::Config::ParameterContainer& parameters )
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
   const long long int elementsPerTest = max_size / sizeof( ElementSize );
   if( ! min_size )
      min_size = TNL::SystemInfo::getCacheLineSize();
   for( size_t size = min_size; size <= max_size; size *= 2 ) {
      benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
         { "element size", TNL::convertToString( ElementSize ) },
         { "array size", TNL::convertToString( size ) },
         { "access type", access_type }
      }));
      benchmark.setDatasetSize( size );
      TestArray< ElementSize > array( size );
      array.setElementsPerTest( elementsPerTest );
      benchmark.setOperationsPerLoop( elementsPerTest );
      std::cerr << "Operations per loop = " << array.getElementsCount() << std::endl;
      if( access_type == "sequential" )
         array.setupSequentialTest();
      else if( access_type == "random" )
         array.setupRandomTest();
      auto compute = [&] () {
         array.performTest();
      };
      benchmark.time< Device >( device, compute );
   }
   return true;
}

template< typename Device >
bool resolveElementSize( const TNL::Config::ParameterContainer& parameters )
{
   int element_size = parameters.getParameter< int >( "element-size" );
   switch( element_size )
   {
      case 1:
         return performBenchmark< Device, 1 >( parameters );
      case 2:
         return performBenchmark< Device, 2 >( parameters );
      case 4:
         return performBenchmark< Device, 4 >( parameters );
      case 8:
         return performBenchmark< Device, 8 >( parameters );
      case 16:
         return performBenchmark< Device, 16 >( parameters );
      case 32:
         return performBenchmark< Device, 32 >( parameters );
      case 64:
         return performBenchmark< Device, 64 >( parameters );
      case 128:
         return performBenchmark< Device, 128 >( parameters );
      case 256:
         return performBenchmark< Device, 256 >( parameters );
   }
   std::cerr << "Element size " << element_size << " is not allowed. It can be only 1, 2, 4, 8, 16, 32, 64, 128, 256." << std::endl;
   return false;
}

int main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( !parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   auto device = parameters.getParameter< TNL::String >( "device" );
   if( device == "sequential" )
      return resolveElementSize< TNL::Devices::Sequential >( parameters ) ? EXIT_SUCCESS : EXIT_FAILURE;
   if( device == "host" )
      return resolveElementSize< TNL::Devices::Host >( parameters ) ? EXIT_SUCCESS : EXIT_FAILURE;
   std::cerr << "Wrong device type " << device << " for the memory access benchmarking, only 'sequential' or 'host' is allowed here." << std::endl;
   return EXIT_FAILURE;
}
