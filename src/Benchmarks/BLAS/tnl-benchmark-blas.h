// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <type_traits>

#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/CudaManaged.h>
#include <TNL/Allocators/HipHost.h>
#include <TNL/Allocators/HipManaged.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/Devices.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>

#include "array-operations.h"
#include "vector-operations.h"
#include "triad.h"

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Device, typename Real >
void
runBlasBenchmarks( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   //   const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   //   const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t minSize = parameters.getParameter< int >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< int >( "max-size" );
   const int sizeStepFactor = parameters.getParameter< int >( "size-step-factor" );
   const std::string deviceName = getDeviceName< Device >();

   // Array operations (there are no sequential implementations)
   if constexpr( ! std::is_same_v< Device, Devices::Sequential > ) {
      std::cout << "\n== Array operations (" << deviceName << ") ==\n\n";
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "host allocator", "Host" },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkArrayOperations< Device, Real >( benchmark, size );
      }
#if defined( __CUDACC__ )
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "host allocator", "CudaHost" },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkArrayOperations< Device, Real, int, Allocators::CudaHost >( benchmark, size );
      }
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "host allocator", "CudaManaged" },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkArrayOperations< Device, Real, int, Allocators::CudaManaged >( benchmark, size );
      }
#elif defined( __HIP__ )
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "host allocator", "HipHost" },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkArrayOperations< Device, Real, int, Allocators::HipHost >( benchmark, size );
      }
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "host allocator", "HipManaged" },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkArrayOperations< Device, Real, int, Allocators::HipManaged >( benchmark, size );
      }
#endif
   }

   // Vector operations
   std::cout << "\n== Vector operations (" << deviceName << ") ==\n\n";
   for( std::size_t size = minSize; size <= maxSize; size *= sizeStepFactor ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "precision", getType< Real >() },
               { "size", convertToString( size ) },
            } ) );
      benchmarkVectorOperations< Device, Real >( benchmark, size );
   }

   // Triad benchmark: copy from host, compute, copy to host
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if constexpr( std::is_same_v< Device, Devices::GPU > ) {
      std::cout << "\n== Triad (" << deviceName << ") ==\n\n";
      for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "precision", getType< Real >() },
                  { "size", convertToString( size ) },
               } ) );
         benchmarkTriad< Real >( benchmark, size );
      }
   }
#endif
}

template< typename Real >
void
resolveDevice( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );

   if( device == "sequential" || device == "all" )
      runBlasBenchmarks< Devices::Sequential, Real >( benchmark, parameters );

   if( device == "host" || device == "all" )
      runBlasBenchmarks< Devices::Host, Real >( benchmark, parameters );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      runBlasBenchmarks< Devices::GPU, Real >( benchmark, parameters );
#endif
}

void
resolvePrecision( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      resolveDevice< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      resolveDevice< double >( benchmark, parameters );
}

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "BLAS benchmark settings:" );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 100000 );
   config.addEntry< int >( "max-size", "Maximum size of arrays/vectors used in the benchmark.", 10000000 );
   config.addEntry< int >(
      "size-step-factor",
      "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and "
      "each following size is stepFactor*previousSize, up to max-size.",
      2 );

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

   if( parameters.getParameter< int >( "size-step-factor" ) <= 1 ) {
      std::cerr << "The value of --size-step-factor must be greater than 1.\n";
      return EXIT_FAILURE;
   }

   if( ! Devices::Host::setup( parameters ) || ! Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolvePrecision( benchmark, parameters );

   return EXIT_SUCCESS;
}
