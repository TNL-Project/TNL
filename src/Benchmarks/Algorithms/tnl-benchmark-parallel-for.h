// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <iostream>
#include <limits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/String.h>

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Parallel for benchmark settings:" );
   config.addEntry< int >( "min-x-size", "Minimum size over x axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-x-size", "Maximum size over x axis used in the benchmark.", 1 << 24 );
   config.addEntry< int >(
      "x-size-step-factor",
      "Factor determining the size grows over x axis. First size is min-x-size and each "
      "following size is x-size-step-factor*previousSize, up to max-x-size.",
      2 );

   config.addEntry< int >( "min-y-size", "Minimum size over y axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-y-size", "Maximum size over y axis used in the benchmark.", 1 << 24 );
   config.addEntry< int >(
      "y-size-step-factor",
      "Factor determining the size grows over y axis. First size is min-y-size and each "
      "following size is y-size-step-factor*previousSize, up to max-y-size.",
      2 );

   config.addEntry< int >( "min-z-size", "Minimum size over z axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-z-size", "Maximum size over z axis used in the benchmark.", 1 << 24 );
   config.addEntry< int >(
      "z-size-step-factor",
      "Factor determining the size grows over z axis. First size is min-z-size and each "
      "following size is z-size-step-factor*previousSize, up to max-z-size.",
      2 );

   config.addEntry< int >(
      "fix-array-size",
      "Measure only the index ranges with this total number of elements, so that the 1D, 2D and 3D "
      "variants can be compared. Zero means measuring all index ranges.",
      1 << 24 );

   config.addEntry< TNL::String >( "device", "Device the benchmark will run on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
}

/**
 * \brief Returns the total number of elements of the index range or -1 if it does not fit into `int`.
 *
 * The vectors used in the benchmark are indexed by `int`, so larger index ranges are skipped.
 */
inline std::int64_t
getTotalSize( std::int64_t xSize, std::int64_t ySize = 1, std::int64_t zSize = 1 )
{
   constexpr std::int64_t maxSize = std::numeric_limits< int >::max();
   if( xSize > maxSize || ySize > maxSize / xSize || zSize > maxSize / ( xSize * ySize ) )
      return -1;
   return xSize * ySize * zSize;
}

/**
 * \brief Checks whether the index range with the given total number of elements should be measured.
 */
inline bool
measureTotalSize( std::int64_t totalSize, std::int64_t fixArraySize )
{
   return totalSize > 0 && ( fixArraySize <= 0 || totalSize == fixArraySize );
}

template< typename Device >
void
runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const std::int64_t min_x_size = parameters.getParameter< int >( "min-x-size" );
   const std::int64_t max_x_size = parameters.getParameter< int >( "max-x-size" );
   const std::int64_t x_size_step_factor = parameters.getParameter< int >( "x-size-step-factor" );
   const std::int64_t min_y_size = parameters.getParameter< int >( "min-y-size" );
   const std::int64_t max_y_size = parameters.getParameter< int >( "max-y-size" );
   const std::int64_t y_size_step_factor = parameters.getParameter< int >( "y-size-step-factor" );
   const std::int64_t min_z_size = parameters.getParameter< int >( "min-z-size" );
   const std::int64_t max_z_size = parameters.getParameter< int >( "max-z-size" );
   const std::int64_t z_size_step_factor = parameters.getParameter< int >( "z-size-step-factor" );
   const std::int64_t fix_array_size = parameters.getParameter< int >( "fix-array-size" );

   if( min_x_size < 1 || min_y_size < 1 || min_z_size < 1 ) {
      std::cerr << "The minimal sizes must be positive.\n";
      return;
   }
   if( x_size_step_factor <= 1 || y_size_step_factor <= 1 || z_size_step_factor <= 1 ) {
      std::cerr << "The size step factors must be greater than 1.\n";
      return;
   }

   // 1D parallelFor
   for( std::int64_t x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor ) {
      const std::int64_t total_size = getTotalSize( x_size );
      if( ! measureTotalSize( total_size, fix_array_size ) )
         continue;
      const int size_x = x_size;
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark::MetadataColumns(
            { { "dimension", TNL::convertToString( 1 ) },
              { "x size", TNL::convertToString( x_size ) },
              { "y size", TNL::convertToString( 1 ) },
              { "z size", TNL::convertToString( 1 ) } } ) );

      benchmark.setDatasetSize( total_size * sizeof( int ) );
      TNL::Containers::Vector< int, Device > v( total_size, 0 );
      auto v_view = v.getView();

      auto reset = [ & ]() mutable
      {
         v_view = 0;
      };
      auto parallelForBenchmark = [ & ]()
      {
         TNL::Algorithms::parallelFor< Device >(
            0,
            size_x,
            [ = ] __cuda_callable__( int i ) mutable
            {
               v_view[ i ] = 1;
            } );
      };
      benchmark.time< Device >( reset, "parallelFor 1D", parallelForBenchmark );
   }

   // 2D parallelFor
   for( std::int64_t x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor )
      for( std::int64_t y_size = min_y_size; y_size <= max_y_size; y_size *= y_size_step_factor ) {
         const std::int64_t total_size = getTotalSize( x_size, y_size );
         if( ! measureTotalSize( total_size, fix_array_size ) )
            continue;
         const int size_x = x_size;
         const int size_y = y_size;
         benchmark.setMetadataColumns(
            TNL::Benchmarks::Benchmark::MetadataColumns(
               { { "dimension", TNL::convertToString( 2 ) },
                 { "x size", TNL::convertToString( x_size ) },
                 { "y size", TNL::convertToString( y_size ) },
                 { "z size", TNL::convertToString( 1 ) } } ) );

         benchmark.setDatasetSize( total_size * sizeof( int ) );
         TNL::Containers::Vector< int, Device > v( total_size, 0 );
         auto v_view = v.getView();

         auto reset = [ & ]() mutable
         {
            v_view = 0;
         };
         auto parallelForBenchmark = [ & ]()
         {
            TNL::Algorithms::parallelFor< Device >(
               TNL::Containers::StaticArray< 2, int >{ 0, 0 },
               TNL::Containers::StaticArray< 2, int >{ size_x, size_y },
               [ = ] __cuda_callable__( const TNL::Containers::StaticArray< 2, int >& i ) mutable
               {
                  v_view[ i.y() * size_x + i.x() ] = 1;
               } );
         };
         benchmark.time< Device >( reset, "parallelFor 2D", parallelForBenchmark );
      }

   // 3D parallelFor
   for( std::int64_t x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor )
      for( std::int64_t y_size = min_y_size; y_size <= max_y_size; y_size *= y_size_step_factor )
         for( std::int64_t z_size = min_z_size; z_size <= max_z_size; z_size *= z_size_step_factor ) {
            const std::int64_t total_size = getTotalSize( x_size, y_size, z_size );
            if( ! measureTotalSize( total_size, fix_array_size ) )
               continue;
            const int size_x = x_size;
            const int size_y = y_size;
            const int size_z = z_size;
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark::MetadataColumns(
                  { { "dimension", TNL::convertToString( 3 ) },
                    { "x size", TNL::convertToString( x_size ) },
                    { "y size", TNL::convertToString( y_size ) },
                    { "z size", TNL::convertToString( z_size ) } } ) );

            benchmark.setDatasetSize( total_size * sizeof( int ) );
            TNL::Containers::Vector< int, Device > v( total_size, 0 );
            auto v_view = v.getView();

            auto reset = [ & ]() mutable
            {
               v_view = 0;
            };
            auto parallelForBenchmark = [ & ]()
            {
               TNL::Algorithms::parallelFor< Device >(
                  TNL::Containers::StaticArray< 3, int >{ 0, 0, 0 },
                  TNL::Containers::StaticArray< 3, int >{ size_x, size_y, size_z },
                  [ = ] __cuda_callable__( const TNL::Containers::StaticArray< 3, int >& i ) mutable
                  {
                     v_view[ ( i.z() * size_y + i.y() ) * size_x + i.x() ] = 1;
                  } );
            };
            benchmark.time< Device >( reset, "parallelFor 3D", parallelForBenchmark );
         }
}

void
resolveDevice( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto device = parameters.getParameter< TNL::String >( "device" );
   if( device == "sequential" || device == "all" )
      runBenchmark< TNL::Devices::Sequential >( benchmark, parameters );
   if( device == "host" || device == "all" )
      runBenchmark< TNL::Devices::Host >( benchmark, parameters );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      runBenchmark< TNL::Devices::GPU >( benchmark, parameters );
#endif
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription config;

   configSetup( config );

   if( ! TNL::Config::parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolveDevice( benchmark, parameters );

   return EXIT_SUCCESS;
}
