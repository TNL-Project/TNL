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

#include <TNL/Benchmarks/Benchmark.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

void
reset()
{}

using index_type = int;

template< typename Device >
void
benchmark_reduction1D( Benchmark& benchmark, index_type size )
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
   benchmark.setMetadataColumns( Benchmark::MetadataColumns( { { "size", convertToString( size ) } } ) );
   benchmark.time< Device >( reset, getDeviceName< Device >(), compute );
}

template< typename Device >
void
benchmark_reduction2D( Benchmark& benchmark, index_type size, index_type n )
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
      Reduction2D< Device >::reduce( static_cast< index_type >( 0 ), fetch, std::plus<>{}, size, n, result.getView() );
   };

   const double datasetSize = ( size * n + n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "2D", datasetSize );
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns( { { "size", convertToString( size ) }, { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, getDeviceName< Device >(), compute );
}

template< typename Device >
void
benchmark_reduction3D( Benchmark& benchmark, index_type size, index_type m, index_type n )
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
      Reduction3D< Device >::reduce( static_cast< index_type >( 0 ), fetch, std::plus<>{}, size, m, n, output );
   };

   const double datasetSize = ( m * n * size + m * n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "3D", datasetSize );
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         { { "size", convertToString( size ) }, { "m", convertToString( m ) }, { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, getDeviceName< Device >(), compute );
}

template< typename Device >
void
run_benchmarks( Benchmark& benchmark )
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
resolveDevice( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< String >( "device" );

   if( device == "sequential" || device == "all" )
      run_benchmarks< Devices::Sequential >( benchmark );

   if( device == "host" || device == "all" )
      run_benchmarks< Devices::Host >( benchmark );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      run_benchmarks< Devices::GPU >( benchmark );
#endif
}

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Reduction benchmark settings:" );
   config.addEntry< String >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

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

   if( ! Devices::Host::setup( parameters ) || ! Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolveDevice( benchmark, parameters );

   return EXIT_SUCCESS;
}
