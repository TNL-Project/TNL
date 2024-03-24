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

#include <TNL/Containers/ndarray/Reduce.h>

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

template< typename T, T... ints >
std::string
print_sequence( std::integer_sequence< T, ints... > int_seq )
{
   return ( ( convertToString( ints ) + " " ) + ... );
}

void
reset()
{}

using index_type = int;

template< typename Device >
void
benchmark_ndarray_reduction1D( Benchmark<>& benchmark, index_type size )
{
   NDArray< index_type, SizesHolder< index_type, 0 >, std::index_sequence< 0 >, Device > input;

   input.setSizes( size );

   input.setValue( 10 );

   auto compute = [ & ]()
   {
      auto res = nd_reduce( input, TNL::Plus{}, 0 );
      (void) res;
   };

   const double datasetSize = size * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "1D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "size", convertToString( size ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device, typename Permutation, std::size_t axis >
void
benchmark_ndarray_reduction2D( Benchmark<>& benchmark, index_type size, index_type n )
{
   NDArray< index_type, SizesHolder< index_type, 0, 0 >, Permutation, Device > input;
   NDArray< index_type, SizesHolder< index_type, 0 >, std::index_sequence< 0 >, Devices::Host > result;

   input.setSizes( size, n );

   input.setValue( 10 );

   auto compute = [ & ]()
   {
      nd_reduce< axis >( input, TNL::Plus{}, 0, result );
   };

   const double datasetSize = ( size * n + n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "2D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "axis", convertToString( axis ) },
                                                                 { "permutation", print_sequence( Permutation{} ) },
                                                                 { "size", convertToString( size ) },
                                                                 { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device, typename Permutation, std::size_t axis >
void
benchmark_ndarray_reduction3D( Benchmark<>& benchmark, index_type size, index_type m, index_type n )
{
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0 >, Permutation, Device > input;
   NDArray< index_type, SizesHolder< index_type, 0, 0 >, std::index_sequence< 0, 1 >, Devices::Host > result;

   input.setSizes( size, m, n );

   input.setValue( 10 );

   auto compute = [ & ]()
   {
      nd_reduce< axis >( input, TNL::Plus{}, 0, result );
   };

   const double datasetSize = ( m * n * size + m * n ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "3D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "axis", convertToString( axis ) },
                                                                 { "permutation", print_sequence( Permutation{} ) },
                                                                 { "size", convertToString( size ) },
                                                                 { "m", convertToString( m ) },
                                                                 { "n", convertToString( n ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device, typename Permutation, std::size_t axis >
void
benchmark_ndarray_reduction4D( Benchmark<>& benchmark, index_type size, index_type m, index_type n, index_type o )
{
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0 >, Permutation, Device > input;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Device > result;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Devices::Host > result_host;

   input.setSizes( size, m, n, o );

   input.setValue( 10 );

   auto input_view = input.getView();

   auto compute = [ & ]()
   {
      nd_reduce< axis >( input_view, TNL::Plus{}, 0, result );

      result_host = result;
   };

   const double datasetSize = ( m * n * o * size + m * n * o ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "4D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "axis", convertToString( axis ) },
                                                                 { "permutation", print_sequence( Permutation{} ) },
                                                                 { "size", convertToString( size ) },
                                                                 { "m", convertToString( m ) },
                                                                 { "n", convertToString( n ) },
                                                                 { "o", convertToString( o ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device, typename Permutation, std::size_t axis >
void
benchmark_ndarray_reduction5D( Benchmark<>& benchmark, index_type size, index_type m, index_type n, index_type o, index_type p )
{
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0, 0 >, Permutation, Device > input;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3 >, Device > result;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3 >, Devices::Host > result_host;

   input.setSizes( size, m, n, o, p );

   input.setValue( 10 );

   auto input_view = input.getView();

   auto compute = [ & ]()
   {
      nd_reduce< axis >( input_view, TNL::Plus{}, 0, result );

      result_host = result;
   };

   const double datasetSize = ( m * n * o * p * size + m * n * o * p ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "5D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "axis", convertToString( axis ) },
                                                                 { "permutation", print_sequence( Permutation{} ) },
                                                                 { "size", convertToString( size ) },
                                                                 { "m", convertToString( m ) },
                                                                 { "n", convertToString( n ) },
                                                                 { "o", convertToString( o ) },
                                                                 { "p", convertToString( p ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device, typename Permutation, std::size_t axis >
void
benchmark_ndarray_reduction6D( Benchmark<>& benchmark,
                               index_type size,
                               index_type m,
                               index_type n,
                               index_type o,
                               index_type p,
                               index_type q )
{
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0, 0, 0 >, Permutation, Device > input;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3, 4 >, Device > result;
   NDArray< index_type, SizesHolder< index_type, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3, 4 >, Devices::Host >
      result_host;

   input.setSizes( size, m, n, o, p, q );

   input.setValue( 10 );

   auto input_view = input.getView();

   auto compute = [ & ]()
   {
      nd_reduce< axis >( input_view, TNL::Plus{}, 0, result );

      result_host = result;
   };

   const double datasetSize = ( m * n * o * p * q * size + m * n * o * p * q ) * sizeof( index_type ) / oneGB;
   benchmark.setOperation( "6D", datasetSize );
   benchmark.setMetadataColumns( Benchmark<>::MetadataColumns( { { "axis", convertToString( axis ) },
                                                                 { "permutation", print_sequence( Permutation{} ) },
                                                                 { "size", convertToString( size ) },
                                                                 { "m", convertToString( m ) },
                                                                 { "n", convertToString( n ) },
                                                                 { "o", convertToString( o ) },
                                                                 { "p", convertToString( p ) },
                                                                 { "q", convertToString( q ) } } ) );
   benchmark.time< Device >( reset, performer< Device >(), compute );
}

template< typename Device >
void
run_benchmarks( Benchmark<>& benchmark )
{
   std::vector sizes_23 = { 64, 256, 1024, 4096, 16384 };

   std::vector sizes_4 = { 4, 16, 128, 256 };

   std::vector sizes_56 = { 2, 16, 128 };

   for( index_type size : { 5000000, 50000000, 500000000 } ) {
      benchmark_ndarray_reduction1D< Device >( benchmark, size );
   }

   TNL::Algorithms::staticFor< std::size_t, 0, 2 >(
      [ & ]( auto axis )
      {
         for( index_type size : sizes_23 ) {
            for( index_type m : sizes_23 ) {
               if( size * m > 3e9 )
                  continue;
               benchmark_ndarray_reduction2D< Device, std::index_sequence< 0, 1 >, axis >( benchmark, size, m );
               benchmark_ndarray_reduction2D< Device, std::index_sequence< 1, 0 >, axis >( benchmark, size, m );
            }
         }
      } );

   TNL::Algorithms::staticFor< std::size_t, 0, 3 >(
      [ & ]( auto axis )
      {
         for( std::size_t size : sizes_23 ) {
            for( std::size_t m : sizes_23 ) {
               for( std::size_t n : sizes_23 ) {
                  if( size * m * n > 3e9 )
                     continue;
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 0, 1, 2 >, axis >( benchmark, size, m, n );
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 0, 2, 1 >, axis >( benchmark, size, m, n );
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 1, 0, 2 >, axis >( benchmark, size, m, n );
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 1, 2, 0 >, axis >( benchmark, size, m, n );
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 2, 1, 0 >, axis >( benchmark, size, m, n );
                  benchmark_ndarray_reduction3D< Device, std::index_sequence< 2, 0, 1 >, axis >( benchmark, size, m, n );
               }
            }
         }
      } );

   TNL::Algorithms::staticFor< std::size_t, 0, 4 >(
      [ & ]( auto axis )
      {
         for( std::size_t size : sizes_4 ) {
            for( std::size_t m : sizes_4 ) {
               for( std::size_t n : sizes_4 ) {
                  for( std::size_t o : sizes_4 ) {
                     if( size * m * n * o < 5e7 )
                        continue;
                     if( size * m * n * o > 2e9 )
                        continue;
                     benchmark_ndarray_reduction4D< Device, std::index_sequence< 0, 1, 2, 3 >, axis >(
                        benchmark, size, m, n, o );
                     benchmark_ndarray_reduction4D< Device, std::index_sequence< 1, 2, 3, 0 >, axis >(
                        benchmark, size, m, n, o );
                     benchmark_ndarray_reduction4D< Device, std::index_sequence< 2, 3, 1, 0 >, axis >(
                        benchmark, size, m, n, o );
                     benchmark_ndarray_reduction4D< Device, std::index_sequence< 3, 0, 1, 2 >, axis >(
                        benchmark, size, m, n, o );
                     benchmark_ndarray_reduction4D< Device, std::index_sequence< 3, 2, 0, 1 >, axis >(
                        benchmark, size, m, n, o );
                  }
               }
            }
         }
      } );

   TNL::Algorithms::staticFor< std::size_t, 0, 5 >(
      [ & ]( auto axis )
      {
         for( std::size_t size : sizes_56 ) {
            for( std::size_t m : sizes_56 ) {
               for( std::size_t n : sizes_56 ) {
                  for( std::size_t o : sizes_56 ) {
                     for( std::size_t p : sizes_56 ) {
                        if( size * m * n * o * p < 5e7 )
                           continue;
                        if( size * m * n * o * p > 3e9 )
                           continue;
                        benchmark_ndarray_reduction5D< Device, std::index_sequence< 0, 1, 2, 3, 4 >, axis >(
                           benchmark, size, m, n, o, p );
                        benchmark_ndarray_reduction5D< Device, std::index_sequence< 1, 2, 3, 4, 0 >, axis >(
                           benchmark, size, m, n, o, p );
                        benchmark_ndarray_reduction5D< Device, std::index_sequence< 2, 3, 4, 0, 1 >, axis >(
                           benchmark, size, m, n, o, p );
                        benchmark_ndarray_reduction5D< Device, std::index_sequence< 3, 4, 0, 1, 2 >, axis >(
                           benchmark, size, m, n, o, p );
                        benchmark_ndarray_reduction5D< Device, std::index_sequence< 4, 0, 1, 2, 3 >, axis >(
                           benchmark, size, m, n, o, p );
                     }
                  }
               }
            }
         }
      } );

   TNL::Algorithms::staticFor< std::size_t, 0, 6 >(
      [ & ]( auto axis )
      {
         for( std::size_t size : sizes_56 ) {
            for( std::size_t m : sizes_56 ) {
               for( std::size_t n : sizes_56 ) {
                  for( std::size_t o : sizes_56 ) {
                     for( std::size_t p : sizes_56 ) {
                        for( std::size_t q : sizes_56 ) {
                           if( size * m * n * o * p * q < 5e7 )
                              continue;
                           if( size * m * n * o * p * q > 3e9 )
                              continue;
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 0, 1, 2, 3, 4, 5 >, axis >(
                              benchmark, size, m, n, o, p, q );
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 1, 2, 3, 4, 5, 0 >, axis >(
                              benchmark, size, m, n, o, p, q );
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 2, 3, 4, 5, 0, 1 >, axis >(
                              benchmark, size, m, n, o, p, q );
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 3, 4, 5, 0, 1, 2 >, axis >(
                              benchmark, size, m, n, o, p, q );
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 4, 5, 0, 1, 2, 3 >, axis >(
                              benchmark, size, m, n, o, p, q );
                           benchmark_ndarray_reduction6D< Device, std::index_sequence< 5, 0, 1, 2, 3, 4 >, axis >(
                              benchmark, size, m, n, o, p, q );
                        }
                     }
                  }
               }
            }
         }
      } );
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-ndarray-reduction.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "cuda" );
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
