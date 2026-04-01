// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

#include <fstream>
#include <functional>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/CUBMergeSort.h>
#include <TNL/Algorithms/Sorting/CUBRadixSort.h>
#include <TNL/Algorithms/Sorting/STLSort.h>
#include <TNL/Algorithms/Sorting/experimental/Quicksort.h>
#include "generators.h"

#if defined( __CUDACC__ )
   #include "ReferenceAlgorithms/CedermanQuicksort.h"
   #include "ReferenceAlgorithms/MancaQuicksort.h"
   #ifdef HAVE_CUDA_SAMPLES
      #include "ReferenceAlgorithms/NvidiaBitonicSort.h"
   #endif
#endif

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Algorithms::Sorting;

void
setupConfig( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Sorting benchmark settings:" );
   config.addEntry< std::size_t >( "size", "Size of the array to sort.", 1 << 20 );
   config.addEntry< String >( "device", "Run benchmarks using given device.", "host" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "all" );
   config.addEntry< String >( "value-type", "Value type to benchmark.", "int" );
   config.addEntryEnum( "int" );
   config.addEntryEnum( "uint" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );
}

template< typename ValueType >
void
runBenchmark( Benchmark& benchmark, std::size_t size, const String& device )
{
   struct DistributionInfo
   {
      const char* name;
      std::function< std::vector< ValueType >( std::size_t, std::uint32_t ) > generator;
   };

   const std::vector< DistributionInfo > distributions = {
      { "random", generateRandom< ValueType > },           { "shuffle", generateShuffle< ValueType > },
      { "sorted", generateSorted< ValueType > },           { "almost-sorted", generateAlmostSorted< ValueType > },
      { "decreasing", generateDecreasing< ValueType > },   { "gaussian", generateGaussian< ValueType > },
      { "bucket", generateBucket< ValueType > },           { "staggered", generateStaggered< ValueType > },
      { "zero-entropy", generateZeroEntropy< ValueType > }
   };

   for( const auto& dist : distributions ) {
      // Create a vector for storing generated values
      const std::vector< ValueType > vec = dist.generator( size, 0 );

      if( device == "host" || device == "all" ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               { { "size", std::to_string( size ) },
                 { "distribution", dist.name },
                 { "value_type", TNL::getType< ValueType >() },
                 { "device", "host" } } ) );
         benchmark.setDatasetSize( size * sizeof( ValueType ) );

         // Create an Array for sorting
         Containers::Array< ValueType, Devices::Host, std::size_t > arr;

         auto reset = [ &vec, &arr ]()
         {
            // Copy the original values to the array
            arr = vec;
         };

         auto sort = [ &arr ]()
         {
            STLSort::sort( arr );
         };

         BenchmarkResult result;
         benchmark.time< Devices::Host >( reset, "STL sort", sort, result );
      }

#ifdef __CUDACC__
      if( device == "cuda" || device == "all" ) {
         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               { { "size", std::to_string( size ) },
                 { "distribution", dist.name },
                 { "value_type", TNL::getType< ValueType >() },
                 { "device", "cuda" } } ) );
         benchmark.setDatasetSize( size * sizeof( ValueType ) );

         // Create an Array for sorting
         Containers::Array< ValueType, Devices::Cuda, std::size_t > arr;

         auto reset = [ &vec, &arr ]()
         {
            // Copy the original values to the array
            arr = vec;
         };

         auto sortBitonic = [ &arr ]()
         {
            BitonicSort::sort( arr );
         };

         BenchmarkResult result;
         benchmark.time< Devices::Cuda >( reset, "bitonic", sortBitonic, result );

         // Verify bitonic sort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "bitonic sort result is not sorted" );

         auto sortQuicksort = [ &arr ]()
         {
            experimental::Quicksort::sort( arr );
         };

         benchmark.time< Devices::Cuda >( reset, "quicksort", sortQuicksort, result );

         // Verify quicksort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "quicksort result is not sorted" );

         auto sortCederman = [ &arr ]()
         {
            CedermanQuicksort::sort( arr );
         };

         benchmark.time< Devices::Cuda >( reset, "CedermanQuicksort", sortCederman, result );

         // Verify Cederman sort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "CedermanQuicksort result is not sorted" );

         auto sortMancaQuicksort = [ &arr ]()
         {
            MancaQuicksort::sort( arr );
         };

         benchmark.time< Devices::Cuda >( reset, "MancaQuicksort", sortMancaQuicksort, result );

         // Verify Manca sort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "MancaQuicksort result is not sorted" );

   #ifdef HAVE_CUDA_SAMPLES
         // NvidiaBitonicSort: supports only `unsigned int` value type and power-of-two sizes >= 1024
         if constexpr( std::is_same_v< ValueType, unsigned int > ) {
            if( TNL::isPow2( size ) && size >= 1024 ) {
               auto sortNvidiaBitonic = [ &arr ]()
               {
                  NvidiaBitonicSort::sort( arr );
               };

               benchmark.time< Devices::Cuda >( reset, "NvidiaBitonicSort", sortNvidiaBitonic, result );

               if( ! Algorithms::isAscending( arr ) )
                  throw std::runtime_error( "NvidiaBitonicSort result is not sorted" );
            }
            else {
               std::cerr << "Skipping NvidiaBitonicSort for size " << size
                         << " because it supports only power-of-two sizes >= 1024\n";
            }
         }
   #endif

         auto sortCUBMergeSort = [ &arr ]()
         {
            CUBMergeSort::sort( arr );
         };

         benchmark.time< Devices::Cuda >( reset, "CUBMergeSort", sortCUBMergeSort, result );

         // Verify CUBMergeSort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "CUBMergeSort result is not sorted" );

         auto sortCUBRadixSort = [ &arr ]()
         {
            auto view = arr.getView();
            CUBRadixSort::sort( view );
         };

         benchmark.time< Devices::Cuda >( reset, "CUBRadixSort", sortCUBRadixSort, result );

         // Verify CUBRadixSort result
         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "CUBRadixSort result is not sorted" );
      }
#endif
   }
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! Devices::Host::setup( parameters ) || ! Devices::Cuda::setup( parameters ) || ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const auto size = parameters.getParameter< std::size_t >( "size" );
   const String& device = parameters.getParameter< String >( "device" );
   const String& valueType = parameters.getParameter< String >( "value-type" );

   Benchmark benchmark;
   benchmark.setup( parameters );

   if( valueType == "int" || valueType == "all" )
      runBenchmark< int >( benchmark, size, device );
   if( valueType == "uint" || valueType == "all" )
      runBenchmark< unsigned int >( benchmark, size, device );
   if( valueType == "double" || valueType == "all" )
      runBenchmark< double >( benchmark, size, device );

   return EXIT_SUCCESS;
}
