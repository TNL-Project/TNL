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
 * done by one kernel launch.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h): the complete scan in
 * GPUScanKernelLauncher::perform when the input is larger than one tile and
 * fits into one CUDA grid, for both ScanPhaseType variants. It is not used
 * when the scan is performed in two phases explicitly (the distributed scan).
 *
 * Global memory traffic is n reads and n writes (plus the small LookbackState
 * array). The tile loads and stores are vectorized (16-byte accesses) for
 * arrays whose value type size divides 16.
 *
 * Reference:
 * D. Merrill, M. Garland, "Single-pass Parallel Prefix Sum with Decoupled
 * Lookback", NVIDIA Technical Report NVR-2016-002, 2016.
 * https://research.nvidia.com/publication/2016-03_Single-pass-Parallel-Prefix-Sum-Decoupled-Lookback
 */

#include <type_traits>

#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>
#include <TNL/Algorithms/detail/GPUScan/GPUTileScan.h>

namespace TNL::Algorithms::detail {

/* Status flags used in the decoupled lookback synchronisation between CUDA
 * blocks. Each block publishes its per-block aggregate first, optionally
 * followed by its prefix once all predecessor aggregates are known. Other
 * blocks spin on these flags to consume the partial results.
 *
 * It is used in GPUScanKernelLookback function.
 */
enum class LookbackStatus : int
{
   Invalid = 0,    // block has not published anything yet
   Aggregate = 1,  // block has published its per-block aggregate only
   Prefix = 2,     // block has published its complete prefix
};

/* Per-block state published in global memory for the decoupled lookback
 * synchronisation. One entry is allocated per CUDA block.
 *
 * The 128-byte alignment matches the GPU cache-line size on recent NVIDIA
 * architectures. It guarantees that no two blocks ever share a cache line,
 * preventing false sharing between concurrent atomicExch/atomicAdd operations
 * issued by independent blocks. Without it, an atomic write from one block
 * would invalidate the cache line also holding another block's state, forcing
 * redundant memory traffic and serialising otherwise independent blocks.
 *
 * It is used in GPUScanKernelLookback function.
 */
template< typename ValueType >
struct alignas( 128 ) LookbackState
{
   int status = static_cast< int >( LookbackStatus::Invalid );
   ValueType aggregate{};  // written once in 5a, never overwritten
   ValueType prefix{};     // written once in 5c, never overwritten
};

#if defined( __CUDACC__ ) || defined( __HIP__ )

// 16-byte vector type used for vectorized loads/stores in the lookback scan
// kernel. int4 is available on all CUDA targets and supports all value types
// whose sizeof divides 16 (1, 2, 4, 8, 16 bytes).
struct alignas( 16 ) Vec4
{
   int x, y, z, w;
};

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
   LookbackState< ValueType >* states )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using TileScan = GPUTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;
   using BlockScan = GPUBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

   __shared__ Backend::Uninitialized< typename TileScan::Storage > tileStorage;
   __shared__ Backend::Uninitialized< ValueType > sharedAggregateStorage;
   __shared__ Backend::Uninitialized< ValueType > sharedPrefixStorage;

   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
   const int elementsInBlock = TNL::min( remainingElements, maxElementsInBlock );

   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   begin += threadOffset;
   outputBegin += threadOffset;

   auto& storage = tileStorage.get();
   auto& sharedAggregate = sharedAggregateStorage.get();
   auto& sharedPrefix = sharedPrefixStorage.get();

   // Phase 1: strided load of the block tile into shared memory; pad the
   // remainder with identity so the last block runs the same code path.
   // Vectorized int4 loads (16 bytes) cut load instructions for contiguous
   // arrays whose value type size divides 16. When the tile origin is not
   // 16-byte aligned (e.g. scan of a sub-array with an unaligned begin), the
   // unaligned prefix and suffix are loaded scalarly and only the aligned
   // middle uses Vec4 loads (with scalar shared stores, since the middle does
   // not start at a 16B boundary in shared memory). Expression templates fall
   // back to the fully scalar path (no getData()). frontPeel is uniform across
   // the block, so no warp divergence.
   {
      constexpr bool canVectorize = ( sizeof( ValueType ) == 1 || sizeof( ValueType ) == 2 || sizeof( ValueType ) == 4
                                      || sizeof( ValueType ) == 8 || sizeof( ValueType ) == 16 )
                                 && 16 % sizeof( ValueType ) == 0 && IsArrayType< InputView >::value;
      if constexpr( canVectorize ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         const int tileOrigin = begin - threadIdx.x;
         const int frontPeel = ( vecWidth - ( tileOrigin % vecWidth ) ) % vecWidth;
         const int actualFrontPeel = frontPeel < elementsInBlock ? frontPeel : elementsInBlock;
         const int alignedVecs = ( elementsInBlock - actualFrontPeel ) / vecWidth;
         const int alignedEnd = actualFrontPeel + alignedVecs * vecWidth;

         // Scalar load: unaligned front peel (0 to vecWidth-1 elements)
         for( int i = threadIdx.x; i < actualFrontPeel; i += blockDim.x )
            storage.data[ i ] = input[ tileOrigin + i ];

         // Vec4 load from aligned global address + scalar shared stores.
         // storage.data[actualFrontPeel] is not 16B aligned when actualFrontPeel
         // is not a multiple of vecWidth, so we store component-by-component.
         if( alignedVecs > 0 ) {
            const Vec4* inputVec = reinterpret_cast< const Vec4* >( input.getData() + tileOrigin + actualFrontPeel );
            for( int v = threadIdx.x; v < alignedVecs; v += blockDim.x ) {
               const Vec4 vec = inputVec[ v ];
               const ValueType* vals = reinterpret_cast< const ValueType* >( &vec );
   #pragma unroll
               for( int c = 0; c < vecWidth; c++ )
                  storage.data[ actualFrontPeel + v * vecWidth + c ] = vals[ c ];
            }
         }

         // Scalar load: unaligned tail (0 to vecWidth-1 elements)
         for( int i = alignedEnd + threadIdx.x; i < elementsInBlock; i += blockDim.x )
            storage.data[ i ] = input[ tileOrigin + i ];

         // Identity padding
         for( int i = elementsInBlock + threadIdx.x; i < maxElementsInBlock; i += blockDim.x )
            storage.data[ i ] = identity;
      }
      else {
         int idx = threadIdx.x;
         while( idx < elementsInBlock ) {
            storage.data[ idx ] = input[ begin ];
            begin += blockDim.x;
            idx += blockDim.x;
         }
         while( idx < maxElementsInBlock ) {
            storage.data[ idx ] = identity;
            idx += blockDim.x;
         }
      }
   }
   __syncthreads();

   // Phase 2: per-thread local reduction over its valuesPerThread chunk.
   const int chunkOffset = threadIdx.x * valuesPerThread;
   ValueType chunkSum = storage.data[ chunkOffset ];
   #pragma unroll
   for( int i = 1; i < valuesPerThread; i++ )
      chunkSum = reduction( chunkSum, storage.data[ chunkOffset + i ] );

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

      // 5a: publish the block aggregate, then flip the status from Invalid to
      // Aggregate so successors can start consuming it.
      states[ blockIdx.x ].aggregate = blockAggregate;
      __threadfence();
      atomicExch( &states[ blockIdx.x ].status, static_cast< int >( LookbackStatus::Aggregate ) );

      // 5b: walk predecessors right-to-left, accumulating their aggregates.
      // Stop as soon as a predecessor advertises Prefix (its prefix field
      // already contains the sum of all blocks up to and including it).
      ValueType prefix = identity;
      for( int pred = blockIdx.x - 1; pred >= 0; pred-- ) {
         int status;
         do {
            status = atomicAdd( &states[ pred ].status, 0 );
         } while( status == static_cast< int >( LookbackStatus::Invalid ) );

         ValueType predValue =
            ( status == static_cast< int >( LookbackStatus::Prefix ) ) ? states[ pred ].prefix : states[ pred ].aggregate;
         prefix = reduction( predValue, prefix );

         if( status == static_cast< int >( LookbackStatus::Prefix ) )
            break;
      }

      // 5c: publish the inclusive prefix (sum up to and including this block)
      // so successors can stop their walk early. sharedPrefix stays exclusive
      // (sum of previous blocks only) to avoid double-counting in Phase 6.
      states[ blockIdx.x ].prefix = reduction( prefix, blockAggregate );
      __threadfence();
      atomicExch( &states[ blockIdx.x ].status, static_cast< int >( LookbackStatus::Prefix ) );

      sharedPrefix = prefix;
   }
   __syncthreads();

   // Phase 6: combine the block-wide prefix with the per-thread exclusive
   // prefix and write the final scan values back into shared memory.
   ValueType value = reduction( exclusivePrefix, sharedPrefix );

   #pragma unroll
   for( int i = 0; i < valuesPerThread; i++ ) {
      const ValueType inputValue = storage.data[ chunkOffset + i ];
      if( scanType == ScanType::Exclusive )
         storage.data[ chunkOffset + i ] = value;
      value = reduction( value, inputValue );
      if( scanType == ScanType::Inclusive )
         storage.data[ chunkOffset + i ] = value;
   }
   __syncthreads();
   // Phase 7: strided store of the scanned tile back to global memory.
   // Mirrors Phase 1: peel front/back scalarly, Vec4 global stores for the
   // aligned middle (with scalar shared reads, since the middle does not
   // start at a 16B boundary in shared memory).
   {
      constexpr bool canVectorize = ( sizeof( ValueType ) == 1 || sizeof( ValueType ) == 2 || sizeof( ValueType ) == 4
                                      || sizeof( ValueType ) == 8 || sizeof( ValueType ) == 16 )
                                 && 16 % sizeof( ValueType ) == 0 && IsArrayType< OutputView >::value;
      if constexpr( canVectorize ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         const int outputTileOrigin = outputBegin - threadIdx.x;
         const int frontPeel = ( vecWidth - ( outputTileOrigin % vecWidth ) ) % vecWidth;
         const int actualFrontPeel = frontPeel < elementsInBlock ? frontPeel : elementsInBlock;
         const int alignedVecs = ( elementsInBlock - actualFrontPeel ) / vecWidth;
         const int alignedEnd = actualFrontPeel + alignedVecs * vecWidth;

         // Scalar store: unaligned front peel
         for( int i = threadIdx.x; i < actualFrontPeel; i += blockDim.x )
            output[ outputTileOrigin + i ] = storage.data[ i ];

         // Scalar shared reads + Vec4 global store: aligned middle
         if( alignedVecs > 0 ) {
            Vec4* outputVec = reinterpret_cast< Vec4* >( output.getData() + outputTileOrigin + actualFrontPeel );
            for( int v = threadIdx.x; v < alignedVecs; v += blockDim.x ) {
               Vec4 vec;
               ValueType* vals = reinterpret_cast< ValueType* >( &vec );
   #pragma unroll
               for( int c = 0; c < vecWidth; c++ )
                  vals[ c ] = storage.data[ actualFrontPeel + v * vecWidth + c ];
               outputVec[ v ] = vec;
            }
         }

         // Scalar store: unaligned tail
         for( int i = alignedEnd + threadIdx.x; i < elementsInBlock; i += blockDim.x )
            output[ outputTileOrigin + i ] = storage.data[ i ];
      }
      else {
         int idx = threadIdx.x;
         while( idx < elementsInBlock ) {
            output[ outputBegin ] = storage.data[ idx ];
            outputBegin += blockDim.x;
            idx += blockDim.x;
         }
      }
   }
#endif
}

}  // namespace TNL::Algorithms::detail
