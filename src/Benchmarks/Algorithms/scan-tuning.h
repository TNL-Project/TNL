// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/detail/GPUScan.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Functional.h>
#include <TNL/TypeInfo.h>

#if defined( __CUDACC__ )
   #include <cub/device/device_scan.cuh>
#endif

namespace TNL::Benchmarks::Scan {

using Algorithms::detail::GPUScanKernelLauncher;
using Algorithms::detail::ScanPhaseType;
using Algorithms::detail::ScanType;

// Configurations of the lookback kernel compiled into the benchmark. The block
// size is given as a multiple of the default block size of the launcher, which
// is 256 * 4 / sizeof(Value) (i.e. 1 kB of values per value of each thread).
using BlockSizeFactors = std::integer_sequence< int, 1, 2 >;
using ValuesPerThreadList = std::integer_sequence< int, 7, 9, 11, 13, 15, 17, 19 >;

template< typename Value >
constexpr int defaultBlockSize = 256 * 4 / sizeof( Value );

template< int... Values >
std::vector< int >
toVector( std::integer_sequence< int, Values... > )
{
   return { Values... };
}

template< ScanType Type >
std::string
scanTypeName()
{
   return Type == ScanType::Inclusive ? "inclusive" : "exclusive";
}

/**
 * \brief Benchmark of the GPU scan for one value type and scan type.
 *
 * For each array size, it measures:
 * - the scan via the public API (performer "TNL", default configuration),
 * - cub::DeviceScan for arithmetic value types in CUDA builds (performer "CUB"),
 * - the lookback kernel for all selected configurations of the block size and
 *   values per thread (performer "TNL lookback").
 *
 * The scan is performed in place on an array of ones. The result of each
 * measurement is compared with the result of the public API.
 */
template< typename Value, ScanType Type >
class ScanTuningBenchmark
{
public:
   using VectorType = Containers::Vector< Value, Devices::GPU, int >;

   ScanTuningBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
   : benchmark( benchmark ),
     blockSizeFactors( parameters.getList< int >( "block-size-factors" ) ),
     valuesPerThread( parameters.getList< int >( "values-per-thread" ) )
   {}

   void
   run( int size )
   {
      this->size = size;
      data.setSize( size );
      reference.setSize( size );
      reference.setValue( Value( 1 ) );
      publicScan( reference );

      // the operation is a metadata element, so it must be set after the metadata columns
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "value type", getType< Value >() },
               { "size", std::to_string( size ) },
               { "block size", "default" },
               { "values per thread", "default" },
            } ) );
      benchmark.setOperation( scanTypeName< Type >() + " scan (inplace)" );
      benchmark.setDatasetSize( 2 * size * sizeof( Value ) );

      // the scan via the public API with the default configuration
      setMetadata( "default", "default" );
      auto computePublic = [ this ]()
      {
         publicScan( data );
      };
      benchmark.time< Devices::GPU >( reset, "TNL", computePublic );
      verify( "TNL", "default", "default" );

#if defined( __CUDACC__ )
      if constexpr( std::is_arithmetic_v< Value > ) {
         std::size_t tempBytes = 0;
         cubScan( nullptr, tempBytes );
         Containers::Array< char, Devices::GPU, std::size_t > temp( tempBytes );
         setMetadata( "-", "-" );
         auto computeCUB = [ & ]()
         {
            cubScan( temp.getData(), tempBytes );
         };
         benchmark.time< Devices::GPU >( reset, "CUB", computeCUB );
         verify( "CUB", "-", "-" );
      }
#endif

      // the lookback kernel with all selected configurations
      benchmarkBlockSizes( BlockSizeFactors{} );
   }

protected:
   void
   publicScan( VectorType& v )
   {
      if constexpr( Type == ScanType::Inclusive )
         Algorithms::inplaceInclusiveScan( v, 0, v.getSize(), TNL::Plus{}, Value( 0 ) );
      else
         Algorithms::inplaceExclusiveScan( v, 0, v.getSize(), TNL::Plus{}, Value( 0 ) );
   }

#if defined( __CUDACC__ )
   void
   cubScan( void* temp, std::size_t& tempBytes )
   {
      if constexpr( Type == ScanType::Inclusive )
         cub::DeviceScan::InclusiveSum( temp, tempBytes, data.getData(), data.getData(), size );
      else
         cub::DeviceScan::ExclusiveSum( temp, tempBytes, data.getData(), data.getData(), size );
   }
#endif

   template< int... Factors >
   void
   benchmarkBlockSizes( std::integer_sequence< int, Factors... > )
   {
      ( benchmarkBlockSize< Factors >( ValuesPerThreadList{} ), ... );
   }

   template< int Factor, int... ValuesPerThread >
   void
   benchmarkBlockSize( std::integer_sequence< int, ValuesPerThread... > )
   {
      constexpr int blockSize = Factor * defaultBlockSize< Value >;
      // skip configurations exceeding the maximal CUDA block size
      if constexpr( blockSize <= 1024 ) {
         if( std::find( blockSizeFactors.begin(), blockSizeFactors.end(), Factor ) == blockSizeFactors.end() )
            return;
         ( benchmarkConfiguration< blockSize, ValuesPerThread >(), ... );
      }
   }

   template< int blockSize, int lookbackValuesPerThread >
   void
   benchmarkConfiguration()
   {
      if( std::find( valuesPerThread.begin(), valuesPerThread.end(), lookbackValuesPerThread ) == valuesPerThread.end() )
         return;
      // valuesPerThread of the other scan kernels is not used by performLookback
      using Launcher =
         GPUScanKernelLauncher< Type, ScanPhaseType::WriteInSecondPhase, Value, blockSize, 7, lookbackValuesPerThread >;

      const std::string blockSizeString = std::to_string( blockSize );
      const std::string valuesPerThreadString = std::to_string( lookbackValuesPerThread );
      setMetadata( blockSizeString, valuesPerThreadString );
      auto compute = [ this ]()
      {
         Launcher::performLookback( data, data, 0, size, 0, TNL::Plus{}, Value( 0 ) );
      };
      benchmark.time< Devices::GPU >( reset, "TNL lookback", compute );
      verify( "TNL lookback", blockSizeString, valuesPerThreadString );
   }

   void
   setMetadata( const std::string& blockSize, const std::string& valuesPerThread )
   {
      benchmark.setMetadataElement( { "block size", blockSize } );
      benchmark.setMetadataElement( { "values per thread", valuesPerThread } );
   }

   // compare the result of the last measured scan with the public API
   void
   verify( const std::string& performer, const std::string& blockSize, const std::string& valuesPerThread )
   {
      // partial sums of ones are exact only up to 2^digits in floating-point types,
      // beyond that the results depend on the order of the operations
      if constexpr( std::is_floating_point_v< Value > ) {
         if( size > ( 1LL << std::numeric_limits< Value >::digits ) )
            return;
      }
      if( data != reference )
         std::cerr << "ERROR: the result of " << performer << " (block size " << blockSize << ", values per thread "
                   << valuesPerThread << ") differs from the result of the public API for size " << size << '\n';
   }

   Benchmark& benchmark;
   const std::vector< int > blockSizeFactors;
   const std::vector< int > valuesPerThread;
   int size = 0;
   VectorType data;
   VectorType reference;
   std::function< void() > reset = [ this ]()
   {
      data.setValue( Value( 1 ) );
   };
};

template< typename Value, ScanType Type >
void
runScanBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // so the sizes are limited to int
   const int minSize = parameters.getParameter< int >( "min-size" );
   const int maxSize = parameters.getParameter< int >( "max-size" );
   const int sizeStepFactor = parameters.getParameter< int >( "size-step-factor" );

   ScanTuningBenchmark< Value, Type > scanBenchmark( benchmark, parameters );
   for( long size = minSize; size <= maxSize; size *= sizeStepFactor )
      scanBenchmark.run( size );
}

}  // namespace TNL::Benchmarks::Scan
