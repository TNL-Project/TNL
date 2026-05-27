// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Containers/NDArray.h>

#include <TNL/Benchmarks/Benchmark.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Containers;
using std::index_sequence;

using value_type = float;
//using index_type = std::size_t;
using index_type = unsigned;

void
reset()
{}

// NOTE: having the sizes as function parameters keeps the compiler from treating them
// as "compile-time constants" and thus e.g. optimizing the 1D iterations with memcpy

template< typename Device >
void
benchmark_1D( Benchmark& benchmark, index_type size = 500000000 )
{
   using ArrayType = NDArray< value_type, SizesHolder< index_type, 0 >, std::make_index_sequence< 1 >, Device >;
   ArrayType a;
   ArrayType b;
   a.setSizes( size );
   b.setSizes( size );
   a.setValue( -1 );
   b.setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [ & ]()
   {
      a.forBoundary(
         [ = ] __cuda_callable__( index_type i ) mutable
         {
            a_view( i ) = b_view( i );
         } );
      a.forInterior(
         [ = ] __cuda_callable__( index_type i ) mutable
         {
            a_view( i ) = b_view( i );
         } );
   };

   const std::size_t datasetSize = 2 * size * sizeof( value_type );
   benchmark.setOperation( "1D" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Device >( reset, "TNL", f );
}

template< typename Device >
void
benchmark_2D( Benchmark& benchmark, index_type size = 22333 )
{
   using ArrayType = NDArray< value_type, SizesHolder< index_type, 0, 0 >, std::make_index_sequence< 2 >, Device >;
   ArrayType a;
   ArrayType b;
   a.setSizes( size, size );
   b.setSizes( size, size );
   a.setValue( -1 );
   b.setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [ & ]()
   {
      a.forBoundary(
         [ = ] __cuda_callable__( index_type i, index_type j ) mutable
         {
            a_view( i, j ) = b_view( i, j );
         } );
      a.forInterior(
         [ = ] __cuda_callable__( index_type i, index_type j ) mutable
         {
            a_view( i, j ) = b_view( i, j );
         } );
   };

   const std::size_t datasetSize = 2 * size * size * sizeof( value_type );
   benchmark.setOperation( "2D" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Device >( reset, "TNL", f );
}

template< typename Device >
void
benchmark_3D( Benchmark& benchmark, index_type size = 800 )
{
   using ArrayType = NDArray< value_type, SizesHolder< index_type, 0, 0, 0 >, std::make_index_sequence< 3 >, Device >;
   ArrayType a;
   ArrayType b;
   a.setSizes( size, size, size );
   b.setSizes( size, size, size );
   a.setValue( -1 );
   b.setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [ & ]()
   {
      a.forBoundary(
         [ = ] __cuda_callable__( index_type i, index_type j, index_type k ) mutable
         {
            a_view( i, j, k ) = b_view( i, j, k );
         } );
      a.forInterior(
         [ = ] __cuda_callable__( index_type i, index_type j, index_type k ) mutable
         {
            a_view( i, j, k ) = b_view( i, j, k );
         } );
   };

   const std::size_t datasetSize = 2 * size * size * size * sizeof( value_type );
   benchmark.setOperation( "3D" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Device >( reset, "TNL", f );
}

// TODO: implement general ParallelBoundaryExecutor
//template< typename Device >
//void benchmark_4D( Benchmark& benchmark, index_type size = 150 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0 >,
//            std::make_index_sequence< 4 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size );
//   b.setSizes( size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j,
//      k, l ) = b_view( i, j, k, l ); } ); a.forInterior( [=] __cuda_callable__ ( index_type i, index_type j, index_type k,
//      index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "4D" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}
//
//template< typename Device >
//void benchmark_5D( Benchmark& benchmark, index_type size = 56 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0 >,
//            std::make_index_sequence< 5 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size );
//   b.setSizes( size, size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable {
//      a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } ); a.forInterior( [=] __cuda_callable__ ( index_type i, index_type
//      j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "5D" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}
//
//template< typename Device >
//void benchmark_6D( Benchmark& benchmark, index_type size = 28 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0, 0 >,
//            std::make_index_sequence< 6 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size, size );
//   b.setSizes( size, size, size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type
//      n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } ); a.forInterior( [=] __cuda_callable__ (
//      index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n
//      ) = b_view( i, j, k, l, m, n ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "6D" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}

template< typename Device >
void
benchmark_2D_perm( Benchmark& benchmark, index_type size = 22333 )
{
   using ArrayType = NDArray< value_type, SizesHolder< index_type, 0, 0 >, std::index_sequence< 1, 0 >, Device >;
   ArrayType a;
   ArrayType b;
   a.setSizes( size, size );
   b.setSizes( size, size );
   a.setValue( -1 );
   b.setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [ & ]()
   {
      a.forBoundary(
         [ = ] __cuda_callable__( index_type i, index_type j ) mutable
         {
            a_view( i, j ) = b_view( i, j );
         } );
      a.forInterior(
         [ = ] __cuda_callable__( index_type i, index_type j ) mutable
         {
            a_view( i, j ) = b_view( i, j );
         } );
   };

   const std::size_t datasetSize = 2 * size * size * sizeof( value_type );
   benchmark.setOperation( "2D permuted" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Device >( reset, "TNL", f );
}

template< typename Device >
void
benchmark_3D_perm( Benchmark& benchmark, index_type size = 800 )
{
   using ArrayType = NDArray< value_type, SizesHolder< index_type, 0, 0, 0 >, std::index_sequence< 2, 1, 0 >, Device >;
   ArrayType a;
   ArrayType b;
   a.setSizes( size, size, size );
   b.setSizes( size, size, size );
   a.setValue( -1 );
   b.setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [ & ]()
   {
      a.forBoundary(
         [ = ] __cuda_callable__( index_type i, index_type j, index_type k ) mutable
         {
            a_view( i, j, k ) = b_view( i, j, k );
         } );
      a.forInterior(
         [ = ] __cuda_callable__( index_type i, index_type j, index_type k ) mutable
         {
            a_view( i, j, k ) = b_view( i, j, k );
         } );
   };

   const std::size_t datasetSize = 2 * size * size * size * sizeof( value_type );
   benchmark.setOperation( "3D permuted" );
   benchmark.setDatasetSize( datasetSize );
   benchmark.time< Device >( reset, "TNL", f );
}

// TODO: implement general ParallelBoundaryExecutor
//template< typename Device >
//void benchmark_4D_perm( Benchmark& benchmark, index_type size = 150 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0 >,
//            std::index_sequence< 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size );
//   b.setSizes( size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j,
//      k, l ) = b_view( i, j, k, l ); } ); a.forInterior( [=] __cuda_callable__ ( index_type i, index_type j, index_type k,
//      index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "4D permuted" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}
//
//template< typename Device >
//void benchmark_5D_perm( Benchmark& benchmark, index_type size = 56 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0 >,
//            std::index_sequence< 4, 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size );
//   b.setSizes( size, size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable {
//      a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } ); a.forInterior( [=] __cuda_callable__ ( index_type i, index_type
//      j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "5D permuted" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}
//
//template< typename Device >
//void benchmark_6D_perm( Benchmark& benchmark, index_type size = 28 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0, 0 >,
//            std::index_sequence< 5, 4, 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size, size );
//   b.setSizes( size, size, size, size, size, size );
//   a.setValue( -1 );
//   b.setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type
//      n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } ); a.forInterior( [=] __cuda_callable__ (
//      index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n
//      ) = b_view( i, j, k, l, m, n ); } );
//   };
//
//   const std::size_t datasetSize = 2 * size * size * size * size * size * size * sizeof(value_type) ;
//   benchmark.setOperation( "6D permuted" );
//   benchmark.setDatasetSize( datasetSize );
//   benchmark.time< Device >( reset, "TNL", f );
//}

template< typename Device >
void
run_benchmarks( Benchmark& benchmark )
{
   benchmark_1D< Device >( benchmark );
   benchmark_2D< Device >( benchmark );
   benchmark_3D< Device >( benchmark );
   //benchmark_4D< Device >( benchmark );
   //benchmark_5D< Device >( benchmark );
   //benchmark_6D< Device >( benchmark );
   benchmark_2D_perm< Device >( benchmark );
   benchmark_3D_perm< Device >( benchmark );
   //benchmark_4D_perm< Device >( benchmark );
   //benchmark_5D_perm< Device >( benchmark );
   //benchmark_6D_perm< Device >( benchmark );
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
   config.addDelimiter( "NDArray benchmark settings:" );
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
