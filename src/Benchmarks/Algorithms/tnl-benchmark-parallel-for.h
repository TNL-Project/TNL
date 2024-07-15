// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/String.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/StaticArray.h>

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< TNL::String >( "log-file", "Name of the log file.", "parallel-for.json" );
   config.addEntry< int >( "loops", "Number of loops to run.", 10 );
   config.addEntry< bool >( "verbose", "Print verbose output.", true );

   config.addEntry< int >( "min-x-size", "Minimum size over x axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-x-size", "Maximum size over x axis used in the benchmark.", 1 << 20 );
   config.addEntry< int >( "x-size-step-factor",
                           "Factor determining the size grows over x axis. First size is min-x-size and each "
                           "following size is x-size-step-factor*previousSize, up to max-x-size.",
                           2 );

   config.addEntry< int >( "min-y-size", "Minimum size over y axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-y-size", "Maximum size over y axis used in the benchmark.", 1 << 20 );
   config.addEntry< int >( "y-size-step-factor",
                           "Factor determining the dimension grows over y axis. First size is min-y-size and each "
                           "following size is y-size-step-factor*previousSize, up to max-y-size.",
                           2 );

   config.addEntry< int >( "min-z-size", "Minimum size over z axis used in the benchmark.", 1 );
   config.addEntry< int >( "max-z-size", "Maximum size over z axis used in the benchmark.", 1 << 20 );
   config.addEntry< int >( "z-size-step-factor",
                           "Factor determining the size grows over z axis. First size is min-z-size and each "
                           "following size is z-size-step-factor*previousSize, up to max-z-size.",
                           2 );

   config.addEntry< int >(
      "fix-array-size", "Perform the benchmark only for the specified size. Zero means performing any benchmark.", 1 << 30 );
   config.addDelimiter( "Device settings:" );
   config.addEntry< TNL::String >( "device", "Device the benchmark will run on.", "all" );
   config.addEntryEnum< TNL::String >( "all" );
   config.addEntryEnum< TNL::String >( "host" );
   config.addEntryEnum< TNL::String >( "sequential" );
   config.addEntryEnum< TNL::String >( "cuda" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );
}

template< typename Device, typename Index = int >
bool
startBenchmark( TNL::Config::ParameterContainer& parameters )
{
   const auto log_file_name = parameters.getParameter< TNL::String >( "log-file" );
   const auto loops = parameters.getParameter< int >( "loops" );
   const auto verbose = parameters.getParameter< bool >( "verbose" );
   const auto min_x_size = parameters.getParameter< int >( "min-x-size" );
   const auto max_x_size = parameters.getParameter< int >( "max-x-size" );
   const auto x_size_step_factor = parameters.getParameter< int >( "x-size-step-factor" );
   const auto min_y_size = parameters.getParameter< int >( "min-y-size" );
   const auto max_y_size = parameters.getParameter< int >( "max-y-size" );
   const auto y_size_step_factor = parameters.getParameter< int >( "y-size-step-factor" );
   const auto min_z_size = parameters.getParameter< int >( "min-z-size" );
   const auto max_z_size = parameters.getParameter< int >( "max-z-size" );
   const auto z_size_step_factor = parameters.getParameter< int >( "z-size-step-factor" );
   const auto fix_array_size = parameters.getParameter< int >( "fix-array-size" );

   std::ofstream log_file( log_file_name, std::ios::out );
   TNL::Benchmarks::Benchmark<> benchmark( log_file, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, log_file_name, ".metadata.json" );

   // 1D parallelFor
   for( Index x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor ) {
      if( fix_array_size > 0 && x_size != fix_array_size )
         continue;
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "dimension", TNL::convertToString( 1 ) },
                                                          { "x-size", TNL::convertToString( x_size ) },
                                                          { "y-size", TNL::convertToString( 1 ) },
                                                          { "z-size", TNL::convertToString( 1 ) } } ) );

      benchmark.setDatasetSize( x_size * sizeof( int ) );
      TNL::Containers::Vector< int, Device > v( x_size, 0.0 );
      auto v_view = v.getView();

      auto reset = [ & ]() mutable
      {
         v_view = 0;
      };
      auto parallelForBenchmark = [ & ]()
      {
         TNL::Algorithms::parallelFor< Device >( 0,
                                                 x_size,
                                                 [ = ] __cuda_callable__( int i ) mutable
                                                 {
                                                    v_view[ i ] = 1;
                                                 } );
      };
      benchmark.time< Device >( reset, "parallelFor 1D", parallelForBenchmark );
   }

   //2D parallelFor
   for( Index x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor )
      for( Index y_size = min_y_size; y_size <= max_y_size; y_size *= y_size_step_factor ) {
         if( fix_array_size > 0 && x_size * y_size != fix_array_size )
            continue;
         benchmark.setMetadataColumns(
            TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "dimension", TNL::convertToString( 1 ) },
                                                             { "x-size", TNL::convertToString( x_size ) },
                                                             { "y-size", TNL::convertToString( y_size ) },
                                                             { "z-size", TNL::convertToString( 1 ) } } ) );

         benchmark.setDatasetSize( x_size * y_size * sizeof( int ) );
         TNL::Containers::Vector< int, Device > v( x_size * y_size, 0.0 );
         auto v_view = v.getView();

         auto reset = [ & ]() mutable
         {
            v_view = 0;
         };
         auto parallelForBenchmark = [ & ]()
         {
            TNL::Algorithms::parallelFor< Device >(
               TNL::Containers::StaticArray< 2, int >{ 0, 0 },
               TNL::Containers::StaticArray< 2, int >{ x_size, y_size },
               [ = ] __cuda_callable__( const TNL::Containers::StaticArray< 2, int >& i ) mutable
               {
                  v_view[ i.y() * x_size + i.x() ] = 1;
               } );
         };
         benchmark.time< Device >( reset, "parallelFor 2D", parallelForBenchmark );
      }

   //3D parallelFor
   for( Index x_size = min_x_size; x_size <= max_x_size; x_size *= x_size_step_factor )
      for( Index y_size = min_y_size; y_size <= max_y_size; y_size *= y_size_step_factor )
         for( Index z_size = min_z_size; z_size <= max_z_size; z_size *= z_size_step_factor ) {
            if( fix_array_size > 0 && x_size * y_size * z_size != fix_array_size )
               continue;
            benchmark.setMetadataColumns(
               TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "dimension", TNL::convertToString( 1 ) },
                                                                { "x-size", TNL::convertToString( x_size ) },
                                                                { "y-size", TNL::convertToString( y_size ) },
                                                                { "z-size", TNL::convertToString( z_size ) } } ) );

            benchmark.setDatasetSize( x_size * y_size * z_size * sizeof( int ) );
            TNL::Containers::Vector< int, Device > v( x_size * y_size * z_size, 0.0 );
            auto v_view = v.getView();

            auto reset = [ & ]() mutable
            {
               v_view = 0;
            };
            auto parallelForBenchmark = [ & ]()
            {
               TNL::Algorithms::parallelFor< Device >(
                  TNL::Containers::StaticArray< 3, int >{ 0, 0, 0 },
                  TNL::Containers::StaticArray< 3, int >{ x_size, y_size, z_size },
                  [ = ] __cuda_callable__( const TNL::Containers::StaticArray< 3, int >& i ) mutable
                  {
                     v_view[ ( i.z() * y_size + i.y() ) * x_size + i.x() ] = 1;
                  } );
            };
            benchmark.time< Device >( reset, "parallelFor 3D", parallelForBenchmark );
         }
   return true;
}

bool
resolveDevice( TNL::Config::ParameterContainer& parameters )
{
   auto device = parameters.getParameter< TNL::String >( "device" );
   if( device == "sequential" || device == "all" )
      return startBenchmark< TNL::Devices::Sequential >( parameters );
   if( device == "host" || device == "all" )
      return startBenchmark< TNL::Devices::Host >( parameters );
   if( device == "cuda" || device == "all" ) {
#ifdef __CUDACC__
      return startBenchmark< TNL::Devices::Cuda >( parameters );
#else
      std::cerr << "The benchmark was not built with CUDA support." << std::endl;
      return false;
#endif
   }
   std::cerr << "Unknown device " << device << "." << std::endl;
   return false;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ConfigDescription config;
   configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! TNL::Config::parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   if( ! resolveDevice( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
