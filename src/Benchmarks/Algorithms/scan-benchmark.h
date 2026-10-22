// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>  // std::partial_sum, std::inclusive_scan, std::exclusive_scan
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/detail/GPUScan.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/Devices.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Functional.h>
#include <TNL/TypeInfo.h>

#include "../ExecutionPolicies.h"
#include "scan-common.h"

#ifdef HAVE_THRUST
   #include <thrust/device_ptr.h>
   #include <thrust/execution_policy.h>
   #include <thrust/scan.h>
#endif

#if defined( __CUDACC__ )
   #include <cub/device/device_scan.cuh>
#elif defined( __HIP__ )
   #include <hipcub/device/device_scan.hpp>
   #include <rocprim/device/device_scan.hpp>
#endif

namespace TNL::Benchmarks::Scan {

using Algorithms::detail::GPUScanKernelLauncher;
using Algorithms::detail::ScanPhaseType;
using Algorithms::detail::ScanType;

#if defined( __HIP__ )
namespace cub = hipcub;
constexpr const char* cubPerformer = "hipCUB";
#else
constexpr const char* cubPerformer = "CUB";
#endif

// the default block size of GPUScanKernelLauncher
template< typename Value >
constexpr int defaultBlockSize = 256 * 4 / sizeof( Value );

template< ScanType Type >
std::string
scanTypeName()
{
   return Type == ScanType::Inclusive ? "inclusive" : "exclusive";
}

// returns true if the list contains the value or "all"
inline bool
isSelected( const std::vector< std::string >& list, const std::string& value )
{
   return std::find( list.begin(), list.end(), value ) != list.end()
       || std::find( list.begin(), list.end(), "all" ) != list.end();
}

/**
 * \brief Benchmark of the scan for one device, value type, and scan type.
 *
 * For each array size, it measures the scan of an array of ones in these
 * variants:
 * - in place (`inplace`),
 * - from one vector to another (`1-vector`),
 * - of the sum of two or three vectors given by an expression template to
 *   another vector (`2-vectors`, `3-vectors`, TNL only).
 *
 * The TNL scan is measured via the public API (performer "TNL"). On the GPU,
 * the in-place scan can also be measured with an explicitly selected kernel
 * and number of values per thread (performers "TNL lookback",
 * "TNL scan-then-propagate", "TNL reduce-then-scan"). The other libraries
 * (the C++ standard library, Thrust, CUB or hipCUB, rocPRIM) are measured for
 * arithmetic value types in the in-place and 1-vector variants.
 *
 * The result of each measurement is compared with a reference computed
 * sequentially on the host.
 */
template< typename Device, typename Value, ScanType Type >
class ScanBenchmark
{
public:
   using VectorType = Containers::Vector< Value, Device, int >;
   using ReferenceVectorType = Containers::Vector< Value, Devices::Sequential, int >;

   ScanBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
   : benchmark( benchmark ),
     variants( parameters.getList< std::string >( "variants" ) ),
     libraries( parameters.getList< std::string >( "libraries" ) ),
     gpuKernels( parameters.getList< std::string >( "gpu-kernels" ) ),
     valuesPerThread( parameters.getList< int >( "values-per-thread" ) )
   {}

   void
   run( int size )
   {
      this->size = size;
      for( auto* v : { &a, &b, &c, &d } )
         v->setSize( size );

      // the operation is a metadata element, so it must be set after the metadata columns
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "value type", getType< Value >() },
               { "size", std::to_string( size ) },
               { "values per thread", "default" },
            } ) );

      if( isSelected( variants, "inplace" ) )
         inplace();
      if( isSelected( variants, "1-vector" ) )
         oneVector();
      if( isSelected( variants, "2-vectors" ) )
         twoVectors();
      if( isSelected( variants, "3-vectors" ) )
         threeVectors();
   }

protected:
   static constexpr bool isGPU = std::is_same_v< Device, Devices::GPU >;

   // the other libraries are measured only for the arithmetic types
   static constexpr bool isArithmetic = std::is_arithmetic_v< Value >;

   void
   inplace()
   {
      setOperation( "inplace", 2, 1 );
      auto reset = [ this ]()
      {
         a.setValue( Value( 1 ) );
      };

      if( isSelected( libraries, "tnl" ) && ( ! isGPU || isSelected( gpuKernels, "default" ) ) ) {
         auto compute = [ this ]()
         {
            if constexpr( Type == ScanType::Inclusive )
               Algorithms::inplaceInclusiveScan( a, 0, size, TNL::Plus{}, Value( 0 ) );
            else
               Algorithms::inplaceExclusiveScan( a, 0, size, TNL::Plus{}, Value( 0 ) );
         };
         time( reset, "TNL", "default", compute, a );
      }

      if constexpr( isArithmetic )
         libraryScans( reset, a, a );

#if defined( __CUDACC__ ) || defined( __HIP__ )
      if constexpr( isGPU )
         if( isSelected( libraries, "tnl" ) )
            benchmarkKernels( ValuesPerThreadList{} );
#endif
   }

   void
   oneVector()
   {
      setOperation( "1 vector", 2, 1 );
      auto reset = [ this ]()
      {
         b.setValue( Value( 0 ) );
      };

      if( isSelected( libraries, "tnl" ) ) {
         auto compute = [ this ]()
         {
            if constexpr( Type == ScanType::Inclusive )
               Algorithms::inclusiveScan( a, b, 0, size, 0, TNL::Plus{}, Value( 0 ) );
            else
               Algorithms::exclusiveScan( a, b, 0, size, 0, TNL::Plus{}, Value( 0 ) );
         };
         time( reset, "TNL", "default", compute, b );
      }

      if constexpr( isArithmetic )
         libraryScans( reset, a, b );
   }

   void
   twoVectors()
   {
      setOperation( "2 vectors", 3, 2 );
      auto reset = [ this ]()
      {
         c.setValue( Value( 0 ) );
      };

      if( isSelected( libraries, "tnl" ) ) {
         auto compute = [ this ]()
         {
            if constexpr( Type == ScanType::Inclusive )
               Algorithms::inclusiveScan( a + b, c, 0, size, 0, TNL::Plus{}, Value( 0 ) );
            else
               Algorithms::exclusiveScan( a + b, c, 0, size, 0, TNL::Plus{}, Value( 0 ) );
         };
         time( reset, "TNL", "default", compute, c );
      }
   }

   void
   threeVectors()
   {
      setOperation( "3 vectors", 4, 3 );
      auto reset = [ this ]()
      {
         d.setValue( Value( 0 ) );
      };

      if( isSelected( libraries, "tnl" ) ) {
         auto compute = [ this ]()
         {
            if constexpr( Type == ScanType::Inclusive )
               Algorithms::inclusiveScan( a + b + c, d, 0, size, 0, TNL::Plus{}, Value( 0 ) );
            else
               Algorithms::exclusiveScan( a + b + c, d, 0, size, 0, TNL::Plus{}, Value( 0 ) );
         };
         time( reset, "TNL", "default", compute, d );
      }
   }

   // Sets the operation and the dataset size (in the number of arrays read and
   // written), resets the input vectors (the previous variants may have
   // overwritten them), and computes the reference result of the scan of an
   // array of the given value.
   void
   setOperation( const std::string& variant, int arrays, int inputValue )
   {
      operation = scanTypeName< Type >() + " scan (" + variant + ")";
      benchmark.setOperation( operation );
      benchmark.setDatasetSize( static_cast< std::size_t >( arrays ) * size * sizeof( Value ) );
      this->inputValue = inputValue;
      for( auto* v : { &a, &b, &c } )
         v->setValue( Value( 1 ) );

      reference.setSize( size );
      Value sum = Value( 0 );
      for( int i = 0; i < size; i++ ) {
         if constexpr( Type == ScanType::Exclusive )
            reference[ i ] = sum;
         sum = sum + Value( inputValue );
         if constexpr( Type == ScanType::Inclusive )
            reference[ i ] = sum;
      }
   }

   // Measures the computation and compares the result with the reference.
   template< typename Reset, typename Compute >
   void
   time( Reset& reset, const std::string& performer, const std::string& valuesPerThread, Compute& compute, VectorType& result )
   {
      benchmark.setMetadataElement( { "values per thread", valuesPerThread } );
      benchmark.time< Device >( reset, performer, compute );
      verify( performer, valuesPerThread, result );
   }

   void
   verify( const std::string& performer, const std::string& valuesPerThread, const VectorType& result )
   {
      // partial sums are exact only up to 2^digits in floating-point types,
      // beyond that the results depend on the order of the operations
      if constexpr( std::is_floating_point_v< Value > ) {
         if( static_cast< long long >( size ) * inputValue > ( 1LL << std::numeric_limits< Value >::digits ) )
            return;
      }
      ReferenceVectorType hostResult;
      hostResult = result;
      if( hostResult != reference )
         std::cerr << "ERROR: the result of " << performer << " (values per thread " << valuesPerThread << ") on "
                   << getDeviceName< Device >() << " differs from the reference for " << operation << ", value type "
                   << getType< Value >() << ", size " << size << '\n';
   }

   // Measures the scans of the other libraries from input to output on Device.
   template< typename Reset >
   void
   libraryScans( Reset& reset, VectorType& input, VectorType& output )
   {
      const Value* in = input.getData();
      Value* out = output.getData();
      const Value* inEnd = in + size;

      if constexpr( std::is_same_v< Device, Devices::Sequential > ) {
         if( Type == ScanType::Inclusive && isSelected( libraries, "std" ) ) {
            auto compute = [ & ]()
            {
               std::partial_sum( in, inEnd, out );
            };
            time( reset, "std::partial_sum", "-", compute, output );
         }
      }
      if constexpr( std::is_same_v< Device, StdAlgorithmsDevice > ) {
         if( isSelected( libraries, "std" ) ) {
            auto compute = [ & ]()
            {
               if constexpr( Type == ScanType::Inclusive )
                  std::inclusive_scan( STDEXEC in, inEnd, out );
               else
                  std::exclusive_scan( STDEXEC in, inEnd, out, Value( 0 ) );
            };
            time( reset, "std::" + scanTypeName< Type >() + "_scan", "-", compute, output );
         }
      }
#ifdef HAVE_THRUST
      if constexpr( std::is_same_v< Device, ThrustHostDevice > ) {
         if( isSelected( libraries, "thrust" ) ) {
            auto compute = [ & ]()
            {
               if constexpr( Type == ScanType::Inclusive )
                  thrust::inclusive_scan( thrust::host, in, inEnd, out );
               else
                  thrust::exclusive_scan( thrust::host, in, inEnd, out );
            };
            time( reset, "thrust::" + scanTypeName< Type >() + "_scan", "-", compute, output );
         }
      }
#endif
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if constexpr( isGPU ) {
   #ifdef HAVE_THRUST
         if( isSelected( libraries, "thrust" ) ) {
            auto compute = [ & ]()
            {
               if constexpr( Type == ScanType::Inclusive )
                  thrust::inclusive_scan(
                     thrust::device,
                     thrust::device_pointer_cast( in ),
                     thrust::device_pointer_cast( inEnd ),
                     thrust::device_pointer_cast( out ) );
               else
                  thrust::exclusive_scan(
                     thrust::device,
                     thrust::device_pointer_cast( in ),
                     thrust::device_pointer_cast( inEnd ),
                     thrust::device_pointer_cast( out ) );
            };
            time( reset, "thrust::" + scanTypeName< Type >() + "_scan", "-", compute, output );
         }
   #endif
         if( isSelected( libraries, "cub" ) ) {
            timeTemporaryStorageScan(
               reset,
               cubPerformer,
               [ & ]( void* temp, std::size_t& tempBytes )
               {
                  if constexpr( Type == ScanType::Inclusive )
                     TNL_BACKEND_SAFE_CALL( cub::DeviceScan::InclusiveSum( temp, tempBytes, in, out, size ) );
                  else
                     TNL_BACKEND_SAFE_CALL( cub::DeviceScan::ExclusiveSum( temp, tempBytes, in, out, size ) );
               },
               output );
         }
   #if defined( __HIP__ )
         if( isSelected( libraries, "rocprim" ) ) {
            timeTemporaryStorageScan(
               reset,
               "rocPRIM",
               [ & ]( void* temp, std::size_t& tempBytes )
               {
                  if constexpr( Type == ScanType::Inclusive )
                     TNL_BACKEND_SAFE_CALL(
                        rocprim::inclusive_scan( temp, tempBytes, in, out, size, rocprim::plus< Value >{} ) );
                  else
                     TNL_BACKEND_SAFE_CALL(
                        rocprim::exclusive_scan( temp, tempBytes, in, out, Value( 0 ), size, rocprim::plus< Value >{} ) );
               },
               output );
         }
   #endif
      }
#endif
   }

#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Measures a library scan which queries the size of its temporary storage
   // when called with nullptr; the storage is allocated outside of the timing.
   template< typename Reset, typename Scan >
   void
   timeTemporaryStorageScan( Reset& reset, const std::string& performer, Scan&& scan, VectorType& output )
   {
      std::size_t tempBytes = 0;
      scan( nullptr, tempBytes );
      Containers::Array< char, Devices::GPU, std::size_t > temp( tempBytes );
      auto compute = [ & ]()
      {
         scan( temp.getData(), tempBytes );
      };
      time( reset, performer, "-", compute, output );
   }

   // the explicitly selected kernels for all selected values per thread
   template< int... ValuesPerThread >
   void
   benchmarkKernels( std::integer_sequence< int, ValuesPerThread... > )
   {
      ( benchmarkKernels< ValuesPerThread >(), ... );
   }

   template< int kernelValuesPerThread >
   void
   benchmarkKernels()
   {
      if( std::find( valuesPerThread.begin(), valuesPerThread.end(), kernelValuesPerThread ) == valuesPerThread.end() )
         return;
      const std::string valuesPerThreadString = std::to_string( kernelValuesPerThread );
      constexpr int blockSize = defaultBlockSize< Value >;
      auto reset = [ this ]()
      {
         a.setValue( Value( 1 ) );
      };

      if( isSelected( gpuKernels, "lookback" ) ) {
         // valuesPerThread of the other scan kernels is not used by performLookback
         using Launcher =
            GPUScanKernelLauncher< Type, ScanPhaseType::WriteInSecondPhase, Value, blockSize, 7, kernelValuesPerThread >;
         auto compute = [ this ]()
         {
            Launcher::performLookback( a, a, 0, size, 0, TNL::Plus{}, Value( 0 ) );
         };
         time( reset, "TNL lookback", valuesPerThreadString, compute, a );
      }
      if( isSelected( gpuKernels, "scan-then-propagate" ) ) {
         using Launcher =
            GPUScanKernelLauncher< Type, ScanPhaseType::WriteInFirstPhase, Value, blockSize, kernelValuesPerThread >;
         auto compute = [ this ]()
         {
            twoPhaseScan< Launcher >();
         };
         time( reset, "TNL scan-then-propagate", valuesPerThreadString, compute, a );
      }
      if( isSelected( gpuKernels, "reduce-then-scan" ) ) {
         using Launcher =
            GPUScanKernelLauncher< Type, ScanPhaseType::WriteInSecondPhase, Value, blockSize, kernelValuesPerThread >;
         auto compute = [ this ]()
         {
            twoPhaseScan< Launcher >();
         };
         time( reset, "TNL reduce-then-scan", valuesPerThreadString, compute, a );
      }
   }

   // the two-phase scan, which is used by the launcher when the input does
   // not fit into one grid
   template< typename Launcher >
   void
   twoPhaseScan()
   {
      const auto blockShifts = Launcher::performFirstPhase( a, a, 0, size, 0, TNL::Plus{}, Value( 0 ) );
      if( blockShifts.getSize() > 2 )
         Launcher::performSecondPhase( a, a, blockShifts, 0, size, 0, TNL::Plus{}, Value( 0 ), Value( 0 ) );
   }
#endif

   Benchmark& benchmark;
   const std::vector< std::string > variants;
   const std::vector< std::string > libraries;
   const std::vector< std::string > gpuKernels;
   const std::vector< int > valuesPerThread;
   int size = 0;
   std::string operation;
   int inputValue = 1;
   VectorType a;
   VectorType b;
   VectorType c;
   VectorType d;
   ReferenceVectorType reference;
};

template< typename Device, typename Value >
void
runScanBenchmarkOnDevice( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // so the sizes are limited to int
   const int minSize = parameters.getParameter< int >( "min-size" );
   const int maxSize = parameters.getParameter< int >( "max-size" );
   const int sizeStepFactor = parameters.getParameter< int >( "size-step-factor" );
   const auto& scanType = parameters.getParameter< std::string >( "scan-type" );

   std::cout << "\n== Scan of " << getType< Value >() << " (" << getDeviceName< Device >() << ") ==\n\n";
   if( scanType == "all" || scanType == "inclusive" ) {
      ScanBenchmark< Device, Value, ScanType::Inclusive > scanBenchmark( benchmark, parameters );
      for( long size = minSize; size <= maxSize; size *= sizeStepFactor )
         scanBenchmark.run( size );
   }
   if( scanType == "all" || scanType == "exclusive" ) {
      ScanBenchmark< Device, Value, ScanType::Exclusive > scanBenchmark( benchmark, parameters );
      for( long size = minSize; size <= maxSize; size *= sizeStepFactor )
         scanBenchmark.run( size );
   }
}

template< typename Value >
void
runScanBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   if( device == "sequential" || device == "all" )
      runScanBenchmarkOnDevice< Devices::Sequential, Value >( benchmark, parameters );
   if( device == "host" || device == "all" )
      runScanBenchmarkOnDevice< Devices::Host, Value >( benchmark, parameters );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      runScanBenchmarkOnDevice< Devices::GPU, Value >( benchmark, parameters );
#endif
}

}  // namespace TNL::Benchmarks::Scan
