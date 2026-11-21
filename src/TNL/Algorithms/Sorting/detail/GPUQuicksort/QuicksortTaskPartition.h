// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>
#include <TNL/Algorithms/detail/PivotPartitionBlock.h>
#include <TNL/Atomic.h>
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

/**
 * \brief Upper bound on the number of tile elements processed by one thread
 * in the first-phase partitioning.
 *
 * It must match the `maxMultiplier` cap in `GPUQuicksort::sort`, which keeps
 * `elementsPerBlock = blockDim * multiplier <= blockDim * maxPerThread` so
 * that a fixed-size register array can hold the tile fragment of one thread.
 */
constexpr int quicksortMaxElementsPerThread = 8;

/*
 * ROCm miscompiles the fused partitioning when it is optimized at -O1 together
 * with the rest of the quicksort: the results are silently corrupted for ranges
 * spanning more than one thread tile and larger inputs abort with an HSA
 * exception. The same code is correct at -O2 and -O3, and at -O1 when this
 * function alone is left unoptimized, so the HIP debug build (compiled with
 * -O1) switches the optimizer off for it. Release builds keep the optimized
 * code. See https://github.com/llvm/llvm-project/issues/245765.
 */
#if defined( __HIP__ ) && ! defined( NDEBUG )
   #pragma clang optimize off
#endif

template< typename Value, typename Index, typename Compare, bool useShared >
__device__
void
partitionTaskBlock(
   Containers::ArrayView< Value, Devices::GPU, Index > src,
   Containers::ArrayView< Value, Devices::GPU, Index > dst,
   const Compare& compare,
   Value* sharedMem,
   const Value& pivot,
   int elementsPerBlock,
   QuicksortTask< Index >& task )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Offset = typename QuicksortTask< Index >::Offset;
   using Algorithms::detail::PivotPartitionEqualPolicy;
   using Algorithms::detail::scanPivotPartitionBlock;
   using Algorithms::detail::scatterPivotPartitionBlock;

   Index myBegin = static_cast< Index >( elementsPerBlock ) * ( blockIdx.x - task.firstBlock );
   Index myEnd = TNL::min( myBegin + elementsPerBlock, src.getSize() );

   auto srcView = src.getView( myBegin, myEnd );

   if constexpr( ! useShared ) {
      // fallback for value types whose tile does not fit into shared memory:
      // two-pass partitioning through the generic pivot partitioning blocks
      static __shared__ Offset smallerStart;
      static __shared__ Offset biggerStart;

      const auto scan = scanPivotPartitionBlock< PivotPartitionEqualPolicy::FillLater, int >( srcView, pivot, compare );

      if( threadIdx.x == blockDim.x - 1 )  // last thread in block has sum of all values
      {
         smallerStart = atomicAdd( &( task.dstBegin ), static_cast< Offset >( scan.inclusive.less ) );
         biggerStart = atomicAdd( &( task.dstEnd ), -static_cast< Offset >( scan.inclusive.greater ) )
                     - static_cast< Offset >( scan.inclusive.greater );
      }
      __syncthreads();

      scatterPivotPartitionBlock< PivotPartitionEqualPolicy::FillLater, false >(
         srcView,
         dst,
         pivot,
         compare,
         scan,
         static_cast< Index >( smallerStart ),
         0,
         static_cast< Index >( biggerStart ),
         sharedMem );
      return;
   }

   /*
    * Fused single-pass partitioning of one tile:
    *
    * 1. every thread loads its tile fragment into registers (one global read
    *    instead of the two passes of scan + scatter),
    * 2. one fused block scan of the packed (less | greater) counters - the
    *    64-bit type dispatches to the shuffle-based GPUBlockScan, so the two
    *    separate shared-memory scans of the generic path become one,
    * 3. the last thread claims the output ranges of this block with two
    *    atomics on the task counters and broadcasts the block totals,
    * 4. the elements are staged in shared memory grouped into [less|greater]
    *    and flushed to the output with coalesced writes.
    */
   constexpr int maxPerThread = quicksortMaxElementsPerThread;
   const int tileCount = static_cast< int >( srcView.getSize() );

   // issue all loads first so that the global read latency of the whole tile
   // fragment overlaps, and classify only after the loads are in flight
   Value vals[ maxPerThread ];
   #pragma unroll
   for( int j = 0; j < maxPerThread; ++j ) {
      const int i = static_cast< int >( threadIdx.x ) + j * static_cast< int >( blockDim.x );
      if( i < tileCount )
         vals[ j ] = srcView[ i ];
   }

   int localLess = 0;
   int localGreater = 0;
   #pragma unroll
   for( int j = 0; j < maxPerThread; ++j ) {
      const int i = static_cast< int >( threadIdx.x ) + j * static_cast< int >( blockDim.x );
      if( i < tileCount ) {
         const Value& v = vals[ j ];
         if( compare( v, pivot ) )
            ++localLess;
         else if( compare( pivot, v ) )
            ++localGreater;
      }
   }

   // low 32 bits = inclusive scan of the less counts, high 32 bits = greater
   using BlockScan =
      Algorithms::detail::GPUBlockScan< Algorithms::detail::ScanType::Inclusive, 0, TNL::Plus, unsigned long long >;
   __shared__ BlockScan::Storage scanStorage;
   __shared__ Offset sharedSmallerStart;
   __shared__ Offset sharedBiggerStart;
   __shared__ int sharedLessTotal;
   __shared__ int sharedGreaterTotal;

   const unsigned long long packed = BlockScan::scan(
      TNL::Plus(),
      0ULL,
      static_cast< unsigned long long >( static_cast< unsigned >( localLess ) )
         | ( static_cast< unsigned long long >( static_cast< unsigned >( localGreater ) ) << 32 ),
      static_cast< int >( threadIdx.x ),
      scanStorage );

   if( threadIdx.x == blockDim.x - 1 ) {
      // the last thread holds the totals of the whole block
      sharedLessTotal = static_cast< int >( packed & 0xffffffffULL );
      sharedGreaterTotal = static_cast< int >( packed >> 32 );
      sharedSmallerStart = atomicAdd( &( task.dstBegin ), static_cast< Offset >( sharedLessTotal ) );
      sharedBiggerStart = atomicAdd( &( task.dstEnd ), -static_cast< Offset >( sharedGreaterTotal ) )
                        - static_cast< Offset >( sharedGreaterTotal );
   }
   __syncthreads();

   // stage the register-cached elements grouped into [less | greater]
   const int exclLess = static_cast< int >( packed & 0xffffffffULL ) - localLess;
   const int exclGreater = static_cast< int >( packed >> 32 ) - localGreater;
   int lessPos = exclLess;
   int greaterPos = sharedLessTotal + exclGreater;
   #pragma unroll
   for( int j = 0; j < maxPerThread; ++j ) {
      const int i = static_cast< int >( threadIdx.x ) + j * static_cast< int >( blockDim.x );
      if( i < tileCount ) {
         const Value& v = vals[ j ];
         if( compare( v, pivot ) )
            sharedMem[ lessPos++ ] = v;
         else if( compare( pivot, v ) )
            sharedMem[ greaterPos++ ] = v;
      }
   }
   __syncthreads();

   // flush the staged groups with coalesced writes
   for( int i = static_cast< int >( threadIdx.x ); i < sharedLessTotal; i += static_cast< int >( blockDim.x ) )
      dst[ static_cast< Index >( sharedSmallerStart ) + i ] = sharedMem[ i ];
   for( int i = static_cast< int >( threadIdx.x ); i < sharedGreaterTotal; i += static_cast< int >( blockDim.x ) )
      dst[ static_cast< Index >( sharedBiggerStart ) + i ] = sharedMem[ static_cast< Index >( sharedLessTotal ) + i ];
#endif
}

#if defined( __HIP__ ) && ! defined( NDEBUG )
   #pragma clang optimize on
#endif

}  // namespace TNL::Algorithms::Sorting::detail
