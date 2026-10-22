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
 * arrays whose value type size divides 16. The LookbackState array is reused
 * across the scans without resetting, the states written by a previous scan
 * are recognized by the epoch stored in the status word.
 *
 * Reference:
 * D. Merrill, M. Garland, "Single-pass Parallel Prefix Sum with Decoupled
 * Lookback", NVIDIA Technical Report NVR-2016-002, 2016.
 * https://research.nvidia.com/publication/2016-03_Single-pass-Parallel-Prefix-Sum-Decoupled-Lookback
 */

#include <cstdint>  // std::uintptr_t
#include <cstring>  // memcpy
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

// 16-byte vector type used for vectorized loads/stores in the lookback scan
// kernel. int4 is available on all CUDA targets and supports all value types
// whose sizeof divides 16 (1, 2, 4, 8, 16 bytes).
struct alignas( 16 ) Vec4
{
   int x, y, z, w;
};

/* Returns the number of elements at the beginning of a tile which must be
 * accessed scalarly, so that the following elements start at a 16-byte
 * boundary and can be accessed via Vec4. The alignment is determined from the
 * address, since the data of a view do not have to start at a 16-byte
 * boundary. If no element starts at a 16-byte boundary (e.g. 16-byte types
 * with 8-byte alignment), the whole tile is accessed scalarly.
 */
template< typename ValueType >
__device__
int
getVec4FrontPeel( const ValueType* tile, int elementsInTile )
{
   const int misalignment = reinterpret_cast< std::uintptr_t >( tile ) % 16;
   if( misalignment == 0 )
      return 0;
   const int peelBytes = 16 - misalignment;
   if( peelBytes % sizeof( ValueType ) != 0 )
      return elementsInTile;
   return TNL::min( static_cast< int >( peelBytes / sizeof( ValueType ) ), elementsInTile );
}

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
   using BlockScan = GPUBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

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

   using InputIndex = typename InputView::IndexType;
   using OutputIndex = typename OutputView::IndexType;

   // the offsets are computed in the index types, since one grid may cover
   // more elements than int can represent
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const InputIndex blockOffset = static_cast< InputIndex >( tileIdx ) * maxElementsInBlock;
   const int elementsInBlock =
      static_cast< int >( TNL::min( end - begin - blockOffset, static_cast< InputIndex >( maxElementsInBlock ) ) );

   begin += blockOffset + threadIdx.x;
   outputBegin += blockOffset + threadIdx.x;

   auto& storage = tileStorage.get();
   auto& sharedAggregate = sharedAggregateStorage.get();
   auto& sharedPrefix = sharedPrefixStorage.get();

   // Phase 1: strided load of the block tile into shared memory; pad the
   // remainder with identity so the last block runs the same code path.
   // Vectorized int4 loads (16 bytes) cut load instructions for contiguous
   // arrays whose value type size divides 16. When the address of the tile
   // origin is not 16-byte aligned (e.g. scan of a sub-array with an unaligned
   // begin or of a view whose data are not aligned), the unaligned prefix and
   // suffix are loaded scalarly and only the aligned middle uses Vec4 loads
   // (with scalar shared stores, since the middle does not start at a 16B
   // boundary in shared memory). Expression templates fall back to the fully
   // scalar path (no getData()). The front peel is uniform across the block,
   // so no warp divergence.
   {
      constexpr bool canVectorize = 16 % sizeof( ValueType ) == 0 && IsArrayType< InputView >::value;
      if constexpr( canVectorize ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         const InputIndex tileOrigin = begin - threadIdx.x;
         const int actualFrontPeel = getVec4FrontPeel( input.getData() + tileOrigin, elementsInBlock );
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
      constexpr bool canVectorize = 16 % sizeof( ValueType ) == 0 && IsArrayType< OutputView >::value;
      if constexpr( canVectorize ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         const OutputIndex outputTileOrigin = outputBegin - threadIdx.x;
         const int actualFrontPeel = getVec4FrontPeel( output.getData() + outputTileOrigin, elementsInBlock );
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
