// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

#include <functional>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/CUBMergeSort.h>
#include <TNL/Algorithms/Sorting/CUBRadixSort.h>
#include <TNL/Algorithms/Sorting/STLSort.h>
#include <TNL/Algorithms/Sorting/experimental/Quicksort.h>
#include "generators.h"

#if defined( __CUDACC__ ) || defined( __HIP__ )
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
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Sorting benchmark settings:" );
   config.addEntry< std::size_t >( "size", "Size of the array to sort.", 1 << 20 );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "value-type", "Value type to benchmark.", "int" );
   config.addEntryEnum( "int" );
   config.addEntryEnum( "uint" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::GPU::configSetup( config );
}

template< typename ValueType >
void
run_benchmark_host( Benchmark& benchmark, const char* distribution, const std::vector< ValueType >& vec )
{
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         { { "size", std::to_string( vec.size() ) },
           { "distribution", distribution },
           { "value type", TNL::getType< ValueType >() },
           { "device", "host" } } ) );
   benchmark.setDatasetSize( vec.size() * sizeof( ValueType ) );

   Containers::Array< ValueType, Devices::Host, std::size_t > arr;

   auto reset = [ &vec, &arr ]()
   {
      arr = vec;
   };

   auto sort = [ &arr ]()
   {
      STLSort::sort( arr );
   };

   BenchmarkResult result;
   benchmark.time< Devices::Host >( reset, "STL sort", sort, result );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
template< typename ValueType >
void
run_benchmark_gpu( Benchmark& benchmark, const char* distribution, const std::vector< ValueType >& vec )
{
   std::string gpuDeviceLabel;
   #if defined( __CUDACC__ )
   gpuDeviceLabel = "cuda";
   #elif defined( __HIP__ )
   gpuDeviceLabel = "hip";
   #else
   gpuDeviceLabel = "gpu";
   #endif
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         { { "size", std::to_string( vec.size() ) },
           { "distribution", distribution },
           { "value type", TNL::getType< ValueType >() },
           { "device", gpuDeviceLabel } } ) );
   benchmark.setDatasetSize( vec.size() * sizeof( ValueType ) );

   Containers::Array< ValueType, Devices::GPU, std::size_t > arr;

   auto reset = [ &vec, &arr ]()
   {
      arr = vec;
   };

   auto sortBitonic = [ &arr ]()
   {
      BitonicSort::sort( arr );
   };

   BenchmarkResult result;
   benchmark.time< Devices::GPU >( reset, "bitonic", sortBitonic, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "bitonic sort result is not sorted" );

   auto sortQuicksort = [ &arr ]()
   {
      experimental::Quicksort::sort( arr );
   };

   benchmark.time< Devices::GPU >( reset, "quicksort", sortQuicksort, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "quicksort result is not sorted" );

   auto sortCederman = [ &arr ]()
   {
      CedermanQuicksort::sort( arr );
   };

   benchmark.time< Devices::GPU >( reset, "CedermanQuicksort", sortCederman, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "CedermanQuicksort result is not sorted" );

   auto sortMancaQuicksort = [ &arr ]()
   {
      MancaQuicksort::sort( arr );
   };

   benchmark.time< Devices::GPU >( reset, "MancaQuicksort", sortMancaQuicksort, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "MancaQuicksort result is not sorted" );

   #ifdef HAVE_CUDA_SAMPLES
   if constexpr( std::is_same_v< ValueType, unsigned int > ) {
      if( TNL::isPow2( vec.size() ) && vec.size() >= 1024UL ) {
         auto sortNvidiaBitonic = [ &arr ]()
         {
            NvidiaBitonicSort::sort( arr );
         };

         benchmark.time< Devices::GPU >( reset, "NvidiaBitonicSort", sortNvidiaBitonic, result );

         if( ! Algorithms::isAscending( arr ) )
            throw std::runtime_error( "NvidiaBitonicSort result is not sorted" );
      }
      else {
         std::cerr << "Skipping NvidiaBitonicSort for size " << vec.size()
                   << " because it supports only power-of-two sizes >= 1024\n";
      }
   }
   #endif

   #if defined( __CUDACC__ )
   auto sortCUBMergeSort = [ &arr ]()
   {
      CUBMergeSort::sort( arr );
   };

   benchmark.time< Devices::GPU >( reset, "CUBMergeSort", sortCUBMergeSort, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "CUBMergeSort result is not sorted" );

   auto sortCUBRadixSort = [ &arr ]()
   {
      auto view = arr.getView();
      Algorithms::Sorting::CUBRadixSort::sort( view );
   };

   benchmark.time< Devices::GPU >( reset, "CUBRadixSort", sortCUBRadixSort, result );

   if( ! Algorithms::isAscending( arr ) )
      throw std::runtime_error( "CUBRadixSort result is not sorted" );
   #endif
}
#endif

template< typename ValueType >
void
run_benchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   const auto size = parameters.getParameter< std::size_t >( "size" );

   struct DistributionInfo
   {
      const char* name;
      std::function< std::vector< ValueType >( std::size_t, std::uint32_t ) > generator;
   };

   const std::vector< DistributionInfo > distributions = {
      { "random", generateRandom< ValueType > },
      { "shuffle", generateShuffle< ValueType > },
      { "sorted", generateSorted< ValueType > },
      { "almost-sorted", generateAlmostSorted< ValueType > },
      { "decreasing", generateDecreasing< ValueType > },
      { "gaussian", generateGaussian< ValueType > },
      { "bucket", generateBucket< ValueType > },
      { "staggered", generateStaggered< ValueType > },
      { "zero-entropy", generateZeroEntropy< ValueType > },
   };

   for( const auto& dist : distributions ) {
      const std::vector< ValueType > vec = dist.generator( size, 0 );

      if( device == "host" || device == "all" )
         run_benchmark_host< ValueType >( benchmark, dist.name, vec );

#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" )
         run_benchmark_gpu< ValueType >( benchmark, dist.name, vec );
#endif
   }
}

void
resolveValueType( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& valueType = parameters.getParameter< std::string >( "value-type" );

   if( valueType == "all" || valueType == "int" )
      run_benchmark< int >( benchmark, parameters );
   if( valueType == "all" || valueType == "uint" )
      run_benchmark< unsigned int >( benchmark, parameters );
   if( valueType == "all" || valueType == "double" )
      run_benchmark< double >( benchmark, parameters );
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

   resolveValueType( benchmark, parameters );

   return EXIT_SUCCESS;
}
