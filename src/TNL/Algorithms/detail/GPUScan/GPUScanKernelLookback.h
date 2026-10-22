// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* GPUScanKernelLookback - single-pass scan with decoupled lookback.
 *
 * Each CUDA block loads its tile of blockSize * valuesPerThread elements into
 * shared memory, computes the per-thread chunk sums and their block-wide scan,
 * and publishes the block aggregate in the LookbackState array. Thread 0 then
 * walks the states of the preceding blocks from right to left, accumulating
 * their aggregates until it reaches a block which has already published its
 * complete prefix. The block publishes its own prefix, so that the succeeding
 * blocks can stop their walk early, and finally scans the tile with the prefix
 * as the initial value and writes the result to the output. The whole scan is
 * done by one kernel launch. The tiles are assigned to the blocks dynamically
 * via an atomic counter in the order in which the blocks start, so that the
 * lookback does not depend on the order in which the blocks are scheduled.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h): the complete scan in
 * GPUScanKernelLauncher::perform when the input is larger than one tile and
 * fits into one CUDA grid, for both ScanPhaseType variants. It is not used
 * when the scan is performed in two phases explicitly (the distributed scan).
 *
 * Global memory traffic is n reads and n writes (plus the small LookbackState
 * array). The tile loads and stores are vectorized (16-byte accesses) for
 * arrays whose value type size divides 16, see GPUTileScan. The LookbackState
 * array is reused across the scans without resetting, the states written by a
 * previous scan are recognized by the epoch stored in the status word.
 *
 * Reference:
 * D. Merrill, M. Garland, "Single-pass Parallel Prefix Sum with Decoupled
 * Lookback", NVIDIA Technical Report NVR-2016-002, 2016.
 * https://research.nvidia.com/publication/2016-03_Single-pass-Parallel-Prefix-Sum-Decoupled-Lookback
 */

#include <cstring>  // memcpy
#include <type_traits>

#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>
#include <TNL/Algorithms/detail/GPUScan/GPUTileScan.h>

namespace TNL::Algorithms::detail {

/* Status flags used in the decoupled lookback synchronisation between CUDA
 * blocks. Each block publishes its per-block aggregate first, optionally
 * followed by its prefix once all predecessor aggregates are known. Other
 * blocks spin on these flags to consume the partial results.
 *
 * The status word of LookbackState stores the flag in the two lowest bits and
 * the epoch (the sequence number of the scan which wrote the state) in the
 * remaining bits, see makeLookbackStatus. A state written by a previous scan
 * has a different epoch, so it is treated as Invalid and the states array
 * does not have to be reset before each scan.
 *
 * It is used in GPUScanKernelLookback function.
 */
enum class LookbackStatus : unsigned int
{
   Invalid = 0,    // block has not published anything yet
   Aggregate = 1,  // block has published its per-block aggregate only
   Prefix = 2,     // block has published its complete prefix
};

// The largest epoch which fits into the status word together with the flag.
constexpr unsigned int maxLookbackEpoch = ~0u >> 2;

__cuda_callable__
constexpr unsigned int
makeLookbackStatus( unsigned int epoch, LookbackStatus flag )
{
   return ( epoch << 2 ) | static_cast< unsigned int >( flag );
}

/* Per-block state published in global memory for the decoupled lookback
 * synchronisation. One entry is allocated per CUDA block.
 *
 * The states of neighbouring blocks are packed densely (16-byte alignment,
 * i.e. 16 bytes per state for 4-byte value types and 32 bytes for 8-byte
 * value types), so that the lookback walk over the preceding blocks touches
 * as few memory sectors as possible. A status word with zero epoch is Invalid
 * for every scan, so a default-constructed state is a valid initial state.
 *
 * It is used in GPUScanKernelLookback function.
 */
template< typename ValueType >
struct alignas( alignof( ValueType ) > 16 ? alignof( ValueType ) : 16 ) LookbackState
{
   ValueType aggregate{};    // written once per scan in 5a
   ValueType prefix{};       // written once per scan in 5c
   unsigned int status = 0;  // makeLookbackStatus( epoch, flag )
};

#if defined( __CUDACC__ ) || defined( __HIP__ )

/* Loads the status word of a lookback state with the acquire semantics: the
 * loads which follow (of the aggregate or prefix published together with the
 * status) cannot be performed before this load. Unlike __threadfence(), the
 * acquire load does not wait for all outstanding memory operations of the
 * thread, which is significantly faster in the lookback loop.
 */
__device__
inline unsigned int
loadAcquire( const unsigned int* ptr )
{
   #if defined( __HIP__ )
   return __hip_atomic_load( ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT );
   #elif defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 700
   unsigned int value;
   asm volatile( "ld.acquire.gpu.global.u32 %0, [%1];" : "=r"( value ) : "l"( ptr ) : "memory" );
   return value;
   #else
   const unsigned int value = *reinterpret_cast< const volatile unsigned int* >( ptr );
   __threadfence();
   return value;
   #endif
}

/* Loads a value written by another CUDA block during the same kernel. The L1
 * cache is not coherent between the multiprocessors, so the load must be
 * volatile to read the value from the L2 cache. Class types are loaded word by
 * word and copied bitwise, the same way as the kernel arguments are passed
 * (the value type does not have to be trivially copyable, but it must be
 * bitwise copyable).
 */
template< typename T >
__device__
T
loadVolatile( const T& ref )
{
   if constexpr( std::is_arithmetic_v< T > )
      return *reinterpret_cast< const volatile T* >( &ref );
   else {
      static_assert(
         sizeof( T ) % sizeof( int ) == 0,
         "loadVolatile supports only arithmetic types and types whose size is a multiple of sizeof(int)" );
      constexpr int words = sizeof( T ) / sizeof( int );
      const volatile int* src = reinterpret_cast< const volatile int* >( &ref );
      alignas( T ) int buffer[ words ];
   #pragma unroll
      for( int i = 0; i < words; i++ )
         buffer[ i ] = src[ i ];
      T result;
      memcpy( static_cast< void* >( &result ), buffer, sizeof( T ) );
      return result;
   }
}

#endif

template<
   ScanType scanType,
   int blockSize,
   int valuesPerThread,
   typename InputView,
   typename OutputView,
   typename Reduction,
   typename ValueType >
__global__
void
GPUScanKernelLookback(
   const InputView input,
   OutputView output,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   typename OutputView::IndexType outputBegin,
   Reduction reduction,
   ValueType identity,
   LookbackState< ValueType >* states,
   unsigned int epoch,
   unsigned long long* tileCounter,
   unsigned long long tileCounterBase )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using TileScan = GPUTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;
   using BlockScan = typename TileScan::BlockScan;

   __shared__ Backend::Uninitialized< typename TileScan::Storage > tileStorage;
   __shared__ Backend::Uninitialized< ValueType > sharedAggregateStorage;
   __shared__ Backend::Uninitialized< ValueType > sharedPrefixStorage;
   __shared__ unsigned int sharedTileIdx;

   // The tiles are assigned to the blocks in the order in which the blocks
   // start, not by blockIdx.x. A block waits for the states of the preceding
   // tiles in the lookback, so the preceding tiles must be assigned to blocks
   // which already run. Otherwise the kernel could deadlock when the blocks
   // with lower blockIdx.x are not scheduled first, which is not guaranteed
   // (e.g. on AMD GPUs).
   if( threadIdx.x == 0 )
      sharedTileIdx = static_cast< unsigned int >( atomicAdd( tileCounter, 1ull ) - tileCounterBase );
   __syncthreads();
   const unsigned int tileIdx = sharedTileIdx;

   // the offset is computed in the index type, since one grid may cover more
   // elements than int can represent
   using InputIndex = typename InputView::IndexType;
   const InputIndex tileOffset = static_cast< InputIndex >( tileIdx ) * TileScan::maxElementsInTile;
   const int elementsInTile =
      static_cast< int >( TNL::min( end - begin - tileOffset, static_cast< InputIndex >( TileScan::maxElementsInTile ) ) );

   auto& storage = tileStorage.get();
   auto& sharedAggregate = sharedAggregateStorage.get();
   auto& sharedPrefix = sharedPrefixStorage.get();

   // Phase 1: load the tile into shared memory; the remainder is padded with
   // identity so the last block runs the same code path.
   TileScan::loadTile( input, begin + tileOffset, elementsInTile, identity, storage );
   __syncthreads();

   // Phase 2: per-thread local reduction over its valuesPerThread chunk.
   const ValueType chunkSum = TileScan::reduceChunk( reduction, storage );

   // Phase 3: cooperative block-wide exclusive scan of the per-thread chunk
   // sums; exclusivePrefix is the sum of all chunks before this thread.
   ValueType exclusivePrefix = BlockScan::scan( reduction, identity, chunkSum, threadIdx.x, storage.blockScanStorage );

   // Phase 4: the last thread publishes the block aggregate (sum of all
   // elements in the block) via shared memory.
   if( threadIdx.x == blockSize - 1 )
      sharedAggregate = reduction( exclusivePrefix, chunkSum );
   __syncthreads();

   // Phase 5: decoupled lookback - performed by thread 0 only.
   if( threadIdx.x == 0 ) {
      ValueType blockAggregate = sharedAggregate;
      const unsigned int aggregateStatus = makeLookbackStatus( epoch, LookbackStatus::Aggregate );
      const unsigned int prefixStatus = makeLookbackStatus( epoch, LookbackStatus::Prefix );

      // 5a: publish the block aggregate, then flip the status from Invalid to
      // Aggregate so successors can start consuming it.
      states[ tileIdx ].aggregate = blockAggregate;
      __threadfence();
      atomicExch( &states[ tileIdx ].status, aggregateStatus );

      // 5b: walk predecessors right-to-left, accumulating their aggregates.
      // Stop as soon as a predecessor advertises Prefix (its prefix field
      // already contains the sum of all blocks up to and including it).
      // States with a different epoch were written by a previous scan and
      // are treated as Invalid.
      ValueType prefix = identity;
      for( int pred = static_cast< int >( tileIdx ) - 1; pred >= 0; pred-- ) {
         unsigned int status;
         do {
            status = loadAcquire( &states[ pred ].status );
         } while( status != aggregateStatus && status != prefixStatus );

         ValueType predValue =
            ( status == prefixStatus ) ? loadVolatile( states[ pred ].prefix ) : loadVolatile( states[ pred ].aggregate );
         prefix = reduction( predValue, prefix );

         if( status == prefixStatus )
            break;
      }

      // 5c: publish the inclusive prefix (sum up to and including this block)
      // so successors can stop their walk early. sharedPrefix stays exclusive
      // (sum of previous blocks only) to avoid double-counting in Phase 6.
      states[ tileIdx ].prefix = reduction( prefix, blockAggregate );
      __threadfence();
      atomicExch( &states[ tileIdx ].status, prefixStatus );

      sharedPrefix = prefix;
   }
   __syncthreads();

   // Phase 6: combine the block-wide prefix with the per-thread exclusive
   // prefix and write the final scan values back into shared memory.
   TileScan::scanChunk( reduction, reduction( exclusivePrefix, sharedPrefix ), storage );
   __syncthreads();

   // Phase 7: store the scanned tile back to global memory.
   TileScan::storeTile( output, outputBegin + tileOffset, elementsInTile, storage );
#endif
}

}  // namespace TNL::Algorithms::detail
