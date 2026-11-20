// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>   // std::count_if, std::partition, std::sort
#include <cstdint>     // std::uint32_t
#include <functional>  // std::function, std::less
#include <iostream>    // std::cerr
#include <stdexcept>   // std::runtime_error
#include <string>      // std::string, std::to_string
#include <vector>      // std::vector

#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/pivotPartition.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>

#include "generators.h"

using namespace TNL;
using namespace TNL::Benchmarks;

void
configSetup( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );
   config.addDelimiter( "Pivot partition benchmark settings:" );
   config.addEntry< std::size_t >( "size", "Size of the array to partition.", 1 << 20 );
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

// copies a host or device array into an STL vector
template< typename Array >
std::vector< typename Array::ValueType >
copyToHost( const Array& array )
{
   std::vector< typename Array::ValueType > result( array.getSize() );
   Algorithms::copy( result, array );
   return result;
}

/* Verifies the result of a three-way partitioning outside of the timed
 * operation:
 *
 * 1. the returned boundaries match the classification counts of the input,
 * 2. every output element satisfies the predicate of its bucket,
 * 3. the output is a permutation of the input, compared via sorted copies.
 *
 * The order of the elements within the buckets is intentionally not checked,
 * since the partitioning is not stable.
 */
template< typename Value >
void
verifyPartition(
   const std::vector< Value >& input,
   const std::vector< Value >& output,
   const Value& pivot,
   std::size_t lessEnd,
   std::size_t greaterBegin,
   const std::string& performer )
{
   const std::less< Value > compare{};
   const std::size_t size = input.size();

   if( output.size() != size )
      throw std::runtime_error( performer + ": the output has a wrong size" );
   if( lessEnd > greaterBegin || greaterBegin > size )
      throw std::runtime_error(
         performer + ": invalid boundaries " + std::to_string( lessEnd ) + ", " + std::to_string( greaterBegin ) );

   // 1. the boundaries must match the classification counts of the input
   const auto lessCount = static_cast< std::size_t >( std::count_if(
      input.begin(),
      input.end(),
      [ & ]( const Value& value )
      {
         return compare( value, pivot );
      } ) );
   const auto greaterCount = static_cast< std::size_t >( std::count_if(
      input.begin(),
      input.end(),
      [ & ]( const Value& value )
      {
         return compare( pivot, value );
      } ) );
   if( lessEnd != lessCount || greaterBegin != size - greaterCount )
      throw std::runtime_error(
         performer + ": wrong returned boundaries " + std::to_string( lessEnd ) + ", " + std::to_string( greaterBegin ) );

   // 2. bucket predicates
   for( std::size_t i = 0; i < lessEnd; i++ )
      if( ! compare( output[ i ], pivot ) )
         throw std::runtime_error(
            performer + ": an element not less than the pivot in the less bucket at index " + std::to_string( i ) );
   for( std::size_t i = lessEnd; i < greaterBegin; i++ )
      if( compare( output[ i ], pivot ) || compare( pivot, output[ i ] ) )
         throw std::runtime_error(
            performer + ": an element not equivalent to the pivot in the equal bucket at index " + std::to_string( i ) );
   for( std::size_t i = greaterBegin; i < size; i++ )
      if( ! compare( pivot, output[ i ] ) )
         throw std::runtime_error(
            performer + ": an element not greater than the pivot in the greater bucket at index " + std::to_string( i ) );

   // 3. permutation check via sorted host copies
   std::vector< Value > inputSorted( input );
   std::vector< Value > outputSorted( output );
   std::sort( inputSorted.begin(), inputSorted.end() );
   std::sort( outputSorted.begin(), outputSorted.end() );
   if( inputSorted != outputSorted )
      throw std::runtime_error( performer + ": the output is not a permutation of the input" );
}

template< typename ValueType >
void
run_benchmark_host( Benchmark& benchmark, const char* distribution, const std::vector< ValueType >& vec )
{
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "size", std::to_string( vec.size() ) },
            { "distribution", distribution },
            { "value type", TNL::getType< ValueType >() },
         } ) );
   benchmark.setDatasetSize( vec.size() * sizeof( ValueType ) );

   // the same input and the same pivot (its middle element) for every performer
   const ValueType pivot = vec[ vec.size() / 2 ];
   const std::less<> compare{};

   Containers::Array< ValueType, Devices::Host, std::size_t > arr;
   // the output of the out-of-place overload is preallocated outside of the
   // timed operation
   Containers::Array< ValueType, Devices::Host, std::size_t > output( vec.size() );

   auto reset = [ &vec, &arr ]()
   {
      arr = vec;
   };

   std::size_t lessEnd = 0;
   std::size_t greaterBegin = 0;

   BenchmarkResult result;

   // reference performer: partition on compare( value, pivot ), then the
   // remainder on ! compare( pivot, value )
   auto partitionStd = [ & ]()
   {
      ValueType* data = arr.getData();
      ValueType* end = data + arr.getSize();
      ValueType* mid = std::partition(
         data,
         end,
         [ & ]( const ValueType& value )
         {
            return compare( value, pivot );
         } );
      ValueType* mid2 = std::partition(
         mid,
         end,
         [ & ]( const ValueType& value )
         {
            return ! compare( pivot, value );
         } );
      lessEnd = static_cast< std::size_t >( mid - data );
      greaterBegin = static_cast< std::size_t >( mid2 - data );
   };

   benchmark.time< Devices::Host >( reset, "std::partition x2", partitionStd, result );
   verifyPartition( vec, copyToHost( arr ), pivot, lessEnd, greaterBegin, "std::partition x2" );

   auto partitionInPlace = [ & ]()
   {
      const auto r = Algorithms::pivotPartition( arr, pivot, compare );
      lessEnd = r.lessEnd;
      greaterBegin = r.greaterBegin;
   };

   benchmark.time< Devices::Host >( reset, "TNL in-place", partitionInPlace, result );
   verifyPartition( vec, copyToHost( arr ), pivot, lessEnd, greaterBegin, "TNL in-place" );

   auto partitionOutOfPlace = [ & ]()
   {
      const auto r = Algorithms::pivotPartition( arr, output, pivot, compare );
      lessEnd = r.lessEnd;
      greaterBegin = r.greaterBegin;
   };

   benchmark.time< Devices::Host >( reset, "TNL out-of-place", partitionOutOfPlace, result );
   verifyPartition( vec, copyToHost( output ), pivot, lessEnd, greaterBegin, "TNL out-of-place" );
   if( copyToHost( arr ) != vec )
      throw std::runtime_error( "TNL out-of-place: the input was modified" );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
template< typename ValueType >
void
run_benchmark_gpu( Benchmark& benchmark, const char* distribution, const std::vector< ValueType >& vec )
{
   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "size", std::to_string( vec.size() ) },
            { "distribution", distribution },
            { "value type", TNL::getType< ValueType >() },
         } ) );
   benchmark.setDatasetSize( vec.size() * sizeof( ValueType ) );

   // the same input and the same pivot (its middle element) for every performer
   const ValueType pivot = vec[ vec.size() / 2 ];
   const std::less<> compare{};

   Containers::Array< ValueType, Devices::GPU, std::size_t > arr;
   // the output of the out-of-place overload is preallocated outside of the
   // timed operation
   Containers::Array< ValueType, Devices::GPU, std::size_t > output( vec.size() );

   auto reset = [ &vec, &arr ]()
   {
      arr = vec;
   };

   std::size_t lessEnd = 0;
   std::size_t greaterBegin = 0;

   BenchmarkResult result;

   // the in-place overload allocates its scratch array inside the timed
   // operation - that is part of its real cost
   auto partitionInPlace = [ & ]()
   {
      const auto r = Algorithms::pivotPartition( arr, pivot, compare );
      lessEnd = r.lessEnd;
      greaterBegin = r.greaterBegin;
   };

   benchmark.time< Devices::GPU >( reset, "TNL in-place", partitionInPlace, result );
   verifyPartition( vec, copyToHost( arr ), pivot, lessEnd, greaterBegin, "TNL in-place" );

   auto partitionOutOfPlace = [ & ]()
   {
      const auto r = Algorithms::pivotPartition( arr, output, pivot, compare );
      lessEnd = r.lessEnd;
      greaterBegin = r.greaterBegin;
   };

   benchmark.time< Devices::GPU >( reset, "TNL out-of-place", partitionOutOfPlace, result );
   verifyPartition( vec, copyToHost( output ), pivot, lessEnd, greaterBegin, "TNL out-of-place" );
   if( copyToHost( arr ) != vec )
      throw std::runtime_error( "TNL out-of-place: the input was modified" );
}
#endif

template< typename ValueType >
void
run_benchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   const auto size = parameters.getParameter< std::size_t >( "size" );

   if( size == 0 ) {
      std::cerr << "The array size must be positive.\n";
      return;
   }

   struct DistributionInfo
   {
      const char* name;
      std::function< std::vector< ValueType >( std::size_t, std::uint32_t ) > generator;
   };

   const std::vector< DistributionInfo > distributions = {
      { "random", generateRandom< ValueType > },
      { "sorted", generateSorted< ValueType > },
      { "decreasing", generateDecreasing< ValueType > },
      { "bucket", generateBucket< ValueType > },
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
