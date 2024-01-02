// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/parseCommandLine.h>
#include "MemoryAccessBenchmarkTestArray.h"
#include "MemoryAccessBenchmark.h"

void
MemoryAccessBenchmark::configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-heat-equation.log" );
   config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntry< int >( "loops", "Number of repetitions for every benchmark test.", 2 );
   config.addEntry< int >( "threads-count", "Number of OpenMP threads for host device.", 1 );
   config.addEntry< int >( "element-size", "Benchmark element size.", 1 );
   config.addEntry< bool >( "read-test", "Read data from the memory.", true );
   config.addEntry< bool >( "write-test", "Write data to the memory.", false );
   config.addEntry< bool >( "central-data-access", "Accessing data in the middle of the testing element.", false );
   config.addEntry< bool >(
      "interleaving", "Sequential access with multiple threads will be interleaved, not in blocks.", false );
   config.addEntry< int >( "min-array-size", "Minimal array size size.", 1 << 10 );
   config.addEntry< int >( "max-array-size", "Maximal array size size.", 1 << 30 );
   config.addEntry< TNL::String >( "access-type", "Type of memory accesses to be benchmarked.", "sequential" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "random" );
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
}

template< int ElementSize >
bool
MemoryAccessBenchmark::performBenchmark( const TNL::Config::ParameterContainer& parameters )
{
   using TestArrayType = MemoryAccessBenchmarkTestArray< ElementSize >;
   using ElementType = typename TestArrayType::ElementType;

   auto output_mode = parameters.getParameter< TNL::String >( "output-mode" );
   auto log_file_name = parameters.getParameter< TNL::String >( "log-file" );
   auto verbose = parameters.getParameter< bool >( "verbose" );
   auto mode = std::ios::out;
   if( output_mode == "append" )
      mode |= std::ios::app;
   std::ofstream log_file( log_file_name.getString(), mode );
   int loops = parameters.getParameter< int >( "loops" );
   TNL::Benchmarks::Benchmark<> benchmark( log_file, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, log_file_name, ".metadata.json" );

   auto access_type = parameters.getParameter< TNL::String >( "access-type" );
   size_t min_size = parameters.getParameter< int >( "min-array-size" );
   size_t max_size = parameters.getParameter< int >( "max-array-size" );
   int threads_count = parameters.getParameter< int >( "threads-count" );
   bool read_test = parameters.getParameter< bool >( "read-test" );
   bool write_test = parameters.getParameter< bool >( "write-test" );
   bool central_data_access = parameters.getParameter< bool >( "central-data-access" );
   bool interleaving = parameters.getParameter< bool >( "interleaving" );

   for( size_t size = min_size; size <= max_size; size *= 2 ) {
      const long long int elementsPerTest = TNL::max( size, 1 << 26 ) / sizeof( ElementType );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "threads", TNL::convertToString( threads_count ) },
                                                          { "access type", access_type },
                                                          { "read test", TNL::convertToString( read_test ) },
                                                          { "write test", TNL::convertToString( write_test ) },
                                                          { "central data", TNL::convertToString( central_data_access ) },
                                                          { "interleaving", TNL::convertToString( interleaving ) },
                                                          { "element size", TNL::convertToString( ElementSize ) },
                                                          { "array size", TNL::convertToString( size ) } } ) );
      TestArrayType array( size );
      array.setThreadsCount( threads_count );
      array.setElementsPerTest( elementsPerTest );
      array.setReadTest( read_test );
      array.setWriteTest( write_test );
      array.setInterleaving( interleaving );
      array.setCentralDataAccess( central_data_access );
      if( access_type == "sequential" )
         array.setupSequentialTest();
      else if( access_type == "random" )
         array.setupRandomTest();
      array.performTest();
      benchmark.setOperationsPerLoop( array.getTestedElementsCountPerThread() );
      double dataset_size = (double) elementsPerTest * sizeof( long int ) / (double) ( 1 << 30 );
      if( read_test || write_test )
         dataset_size *= 2;
      benchmark.setDatasetSize( dataset_size );
      auto compute = [ & ]()
      {
         array.performTest();
      };
      benchmark.time< TNL::Devices::Host >( "host", compute );
   }
   return true;
}
