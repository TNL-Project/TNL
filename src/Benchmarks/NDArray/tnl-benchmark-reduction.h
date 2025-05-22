// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/Reduction2D.h>
#include <TNL/Algorithms/Reduction3D.h>

#include <TNL/Benchmarks/Benchmarks.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
const char*
performer()
{
   if( std::is_same_v< Device, Devices::Host > )
      return "CPU";
   else if( std::is_same_v< Device, Devices::Cuda > )
      return "GPU";
   else
      return "unknown";
}

void
reset()
{}

using index_type = int;

template< typename Device >
void
benchmark_reduction1D( Benchmark<>& benchmark, index_type size )
{
   Array< index_type, Device, index_type > v( size );
   v.setValue( 10 );

   auto compute = [ & ]()
   {
      auto res = reduce< Device >( 0, size, v.getConstView(), TNL::Plus{} );
      (void) res;
   };

   const double datasetSize = size * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "1D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "size", convertToString( size ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device >
void
benchmark_reduction2D( Benchmark<>& benchmark, index_type size, index_type n )
{
   Vector< index_type, Device > v( size * n );
   Vector< index_type, Devices::Sequential > result( n );

   v.setValue( 10 );

   auto v_view = v.getView();

   auto compute = [ & ]()
   {
      auto fetch = [ = ] __cuda_callable__( index_type i, int k )
      {
         TNL_ASSERT_LT( i, size, "fetcher got invalid index i" );
         TNL_ASSERT_LT( k, n, "fetcher got invalid index k" );
         return v_view[ i + k * size ];
      };
      Reduction2D< Device >::reduce( (index_type) 0, fetch, std::plus<>{}, size, n, result.getView() );
   };

   const double datasetSize = ( size * n + n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "2D", datasetSize );
   benchmark.setMetadataColumns(
      Benchmark<>::MetadataColumns( { { "size", convertToString( size ) }, { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device >
void
benchmark_reduction3D( Benchmark<>& benchmark, index_type size, index_type m, index_type n )
{
   Vector< index_type, Device > v( m * n * size );
   Vector< index_type, Devices::Host > result( m * n );

   v.setValue( 10 );

   auto v_view = v.getView();
   auto result_view = result.getView();

   auto compute = [ & ]()
   {
      auto fetch = [ = ] __cuda_callable__( index_type i, index_type k, index_type l )
      {
         TNL_ASSERT_LT( i, size, "fetcher got invalid index i" );
         TNL_ASSERT_LT( k, m, "fetcher got invalid index k" );
         TNL_ASSERT_LT( l, n, "fetcher got invalid index l" );
         return v_view[ i + k * size * n + l * size ];
      };
      auto output = [ = ] __cuda_callable__( index_type k, index_type l ) mutable -> index_type&
      {
         return result_view[ k * n + l ];
      };
      Reduction3D< Device >::reduce( (index_type) 0, fetch, std::plus<>{}, size, m, n, output );
   };

   const double datasetSize = ( m * n * size + m * n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "3D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns(
      { { "size", convertToString( size ) }, { "m", convertToString( m ) }, { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device >
void
run_benchmarks( Benchmark<>& benchmark )
{
   for( index_type size : { 5000000, 500000000 } ) {
      benchmark_reduction1D< Device >( benchmark, size );
   }

   for( index_type size : { 22333, 44666 } ) {
      for( index_type m : { 22330, 11111, 5000, 2000, 1000 } ) {
         benchmark_reduction2D< Device >( benchmark, size, m );
      }
   }

   std::vector sizes = { 50, 64, 96, 150, 200, 400, 800, 1600, 3200, 6400 };
   for( std::size_t size : sizes ) {
      for( std::size_t m : sizes ) {
         for( std::size_t n : sizes ) {
            if( size * m * n > 1e9 )
               continue;
            benchmark_reduction3D< Device >( benchmark, size, m, n );
         }
      }
   }
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-reduction.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
#ifdef __CUDACC__
   config.addEntryEnum( "cuda" );
#endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) || ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String& logFileName = parameters.getParameter< String >( "log-file" );
   const String& outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   const String devices = parameters.getParameter< String >( "devices" );
   if( devices == "all" || devices == "host" )
      run_benchmarks< Devices::Host >( benchmark );
#ifdef __CUDACC__
   if( devices == "all" || devices == "cuda" )
      run_benchmarks< Devices::Cuda >( benchmark );
#endif

   return EXIT_SUCCESS;
}
