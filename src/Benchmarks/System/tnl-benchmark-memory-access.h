// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Benchmarks/Benchmark.h>
#include "MemoryAccessBenchmarkTestArray.h"

#include <stdexcept>

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );
   config.addDelimiter( "Memory access benchmark settings:" );
   config.addEntry< int >( "threads-count", "Number of OpenMP threads for host device.", 1 );
   config.addEntry< int >( "element-size", "Benchmark element size.", 1 );
   config.addEntry< TNL::String >( "test-type", "Type of memory operation to benchmark.", "read" );
   config.addEntryEnum( "read" );
   config.addEntryEnum( "write" );
   config.addEntryEnum( "read-write" );
   config.addEntry< bool >( "central-data-access", "Accessing data in the middle of the testing element.", false );
   config.addEntry< int >( "min-array-size", "Minimal array size size.", 1 << 10 );
   config.addEntry< int >( "max-array-size", "Maximal array size size.", 1 << 30 );
   config.addEntry< TNL::String >( "access-type", "Type of memory accesses to be benchmarked.", "sequential" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "interleaved" );
   config.addEntryEnum( "random" );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
}

template< int ElementSize >
bool
performBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   using TestArrayType = MemoryAccessBenchmarkTestArray< ElementSize >;
   using ElementType = typename TestArrayType::ElementType;

   auto access_type = parameters.getParameter< TNL::String >( "access-type" );
   auto test_type = parameters.getParameter< TNL::String >( "test-type" );
   bool read_test = ( test_type == "read" || test_type == "read-write" );
   bool write_test = ( test_type == "write" || test_type == "read-write" );
   size_t min_size = parameters.getParameter< int >( "min-array-size" );
   size_t max_size = parameters.getParameter< int >( "max-array-size" );
   int threads_count = parameters.getParameter< int >( "threads-count" );
   bool central_data_access = parameters.getParameter< bool >( "central-data-access" );

   for( size_t size = min_size; size <= max_size; size *= 2 ) {
      const long long int elementsPerTest = TNL::max( size, 1 << 26 ) / sizeof( ElementType );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark::MetadataColumns(
            { { "threads", TNL::convertToString( threads_count ) },
              { "access type", access_type },
              { "test type", test_type },
              { "central data", TNL::convertToString( central_data_access ) },
              { "element size", TNL::convertToString( ElementSize ) },
              { "array size", TNL::convertToString( size ) } } ) );

      TestArrayType array( size );
      array.setThreadsCount( threads_count );
      array.setElementsPerTest( elementsPerTest );
      array.setReadTest( read_test );
      array.setWriteTest( write_test );
      array.setCentralDataAccess( central_data_access );

      try {
         if( access_type == "sequential" )
            array.setupSequentialTest( false );
         else if( access_type == "interleaved" )
            array.setupSequentialTest( true );
         else if( access_type == "random" )
            array.setupRandomTest();
         array.performTest();
      }
      catch( const std::runtime_error& e ) {
         std::cerr << "Skipping array size " << size << ": " << e.what() << '\n';
         continue;
      }

      benchmark.setOperationsPerLoop( array.getTestedElementsCountPerThread() );
      double dataset_size = elementsPerTest * sizeof( long int ) / static_cast< double >( 1 << 30 );
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

bool
resolveElementSize( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   int element_size = parameters.getParameter< int >( "element-size" );
   switch( element_size ) {
      case 1:
         return performBenchmark< 1 >( benchmark, parameters );
      case 2:
         return performBenchmark< 2 >( benchmark, parameters );
      case 4:
         return performBenchmark< 4 >( benchmark, parameters );
      case 8:
         return performBenchmark< 8 >( benchmark, parameters );
      case 16:
         return performBenchmark< 16 >( benchmark, parameters );
      case 32:
         return performBenchmark< 32 >( benchmark, parameters );
      case 64:
         return performBenchmark< 64 >( benchmark, parameters );
      case 128:
         return performBenchmark< 128 >( benchmark, parameters );
      case 256:
         return performBenchmark< 256 >( benchmark, parameters );
   }
   std::cerr << "Element size " << element_size << " is not allowed. It can be only 1, 2, 4, 8, 16, 32, 64, 128, 256.\n";
   return false;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   const bool status = resolveElementSize( benchmark, parameters );
   return static_cast< int >( ! status );
}
