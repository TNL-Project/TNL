// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Containers/Array.h>
#include <TNL/TypeTraits.h>
#include "ScanType.h"

namespace TNL::Algorithms::detail {

/* Status flags used in the decoupled lookback synchronisation between CUDA
 * blocks. Each block publishes its per-block aggregate first, optionally
 * followed by its prefix once all predecessor aggregates are known. Other
 * blocks spin on these flags to consume the partial results.
 *
 * It is used in CudaScanKernelLookback function.
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
 * It is used in CudaScanKernelLookback function.
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

// Keeping the following storage structures outside of CudaScan objects make
// them independent of the scan operation type. This allows to reuse the same
// storage for both inclusive and exclusive scan.
template< typename ValueType, int BlockSize >
struct CudaScanStorage
{
   // accessed via Backend::getInterleaving()
   ValueType chunkResults[ BlockSize + BlockSize / Backend::getNumberOfSharedMemoryBanks() ];
   ValueType warpResults[ Backend::getWarpSize() ];
};

template< typename ValueType >
struct CudaScanShflStorage
{
   ValueType warpResults[ Backend::getWarpSize() ];
};

template< typename ValueType, typename BlockStorage, int BlockSize, int ValuesPerThread >
struct CudaTileScanStorage
{
   alignas( 16 ) ValueType data[ BlockSize * ValuesPerThread ];
   BlockStorage blockScanStorage;
};

/* Template for cooperative scan across the CUDA block of threads.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< ScanType scanType, int blockSize, typename Reduction, typename ValueType >
struct CudaBlockScan
{
   // storage to be allocated in shared memory
   using Storage = CudaScanStorage< ValueType, blockSize >;

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__
   static ValueType
   scan( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
   {
      // verify the configuration
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaBlockScan::scan" );
      static_assert(
         blockSize / Backend::getWarpSize() <= Backend::getWarpSize(),
         "blockSize is too large, it would not be possible to scan warpResults using one warp" );

      // store the threadValue in the shared memory
      const int chunkResultIdx = Backend::getInterleaving( tid );
      storage.chunkResults[ chunkResultIdx ] = threadValue;
      __syncthreads();

      // perform the parallel scan on chunkResults inside warps
      const int lane_id = tid % Backend::getWarpSize();
      const int warp_id = tid / Backend::getWarpSize();
      auto warp = cg::tiled_partition< Backend::getWarpSize() >( cg::this_thread_block() );
   #pragma unroll
      for( int stride = 1; stride < Backend::getWarpSize(); stride *= 2 ) {
         ValueType result;
         if( lane_id >= stride )
            result = reduction(
               storage.chunkResults[ chunkResultIdx ], storage.chunkResults[ Backend::getInterleaving( tid - stride ) ] );
         // We must sync all threads in a warp after read and before write to avoid race condition
         warp.sync();
         if( lane_id >= stride )
            storage.chunkResults[ chunkResultIdx ] = result;
         warp.sync();
      }
      threadValue = storage.chunkResults[ chunkResultIdx ];

      // the last thread in warp stores the intermediate result in warpResults
      if( lane_id == Backend::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = threadValue;
      __syncthreads();

      // perform the scan of warpResults using one warp
      if( warp_id == 0 ) {
   #pragma unroll
         for( int stride = 1; stride < blockSize / Backend::getWarpSize(); stride *= 2 ) {
            ValueType result;
            if( lane_id >= stride )
               result = reduction( storage.warpResults[ tid ], storage.warpResults[ tid - stride ] );
            // We must sync all threads in a warp after read and before write to avoid race condition
            warp.sync();
            if( lane_id >= stride )
               storage.warpResults[ tid ] = result;
            warp.sync();
         }
      }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      // shift the result for exclusive scan
      if( scanType == ScanType::Exclusive ) {
         storage.chunkResults[ chunkResultIdx ] = threadValue;
         __syncthreads();
         threadValue = ( tid == 0 ) ? identity : storage.chunkResults[ Backend::getInterleaving( tid - 1 ) ];
      }

      __syncthreads();
      return threadValue;
   }
};

template<
   ScanType scanType,
   int __unused,  // the __shfl implementation does not depend on the blockSize
   typename Reduction,
   typename ValueType >
struct CudaBlockScanShfl
{
   // storage to be allocated in shared memory
   using Storage = CudaScanShflStorage< ValueType >;

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__
   static ValueType
   scan( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
   {
      const int lane_id = tid % Backend::getWarpSize();
      const int warp_id = tid / Backend::getWarpSize();

      // perform the parallel scan across warps
      ValueType total;
      threadValue = warpScan< scanType >( reduction, identity, threadValue, lane_id, total );

      // the last thread in warp stores the result of inclusive scan in warpResults
      if( lane_id == Backend::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = total;
      __syncthreads();

      // the first warp performs the scan of warpResults
      if( warp_id == 0 ) {
         // read from shared memory only if that warp existed
         if( tid < blockDim.x / Backend::getWarpSize() )
            total = storage.warpResults[ lane_id ];
         else
            total = identity;
         storage.warpResults[ lane_id ] = warpScan< ScanType::Inclusive >( reduction, identity, total, lane_id, total );
      }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      __syncthreads();
      return threadValue;
   }

   /* Helper function.
    * Cooperative scan across the warp - each thread will get the result of the
    * scan according to its ID.
    * return value = thread's result of the *warpScanType* scan
    * total = thread's result of the *inclusive* scan
    */
   template< ScanType warpScanType >
   __device__
   static ValueType
   warpScan( const Reduction& reduction, ValueType identity, ValueType threadValue, int lane_id, ValueType& total )
   {
   // perform an inclusive scan
   #pragma unroll
      for( int stride = 1; stride < Backend::getWarpSize(); stride *= 2 ) {
         const ValueType otherValue = Backend::warp_shuffle_up( threadValue, stride );
         if( lane_id >= stride )
            threadValue = reduction( threadValue, otherValue );
      }

      // set the result of the inclusive scan
      total = threadValue;

      // shift the result for exclusive scan
      if( warpScanType == ScanType::Exclusive ) {
         threadValue = Backend::warp_shuffle_up( threadValue, 1 );
         if( lane_id == 0 )
            threadValue = identity;
      }

      return threadValue;
   }
};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, int > : public CudaBlockScanShfl< scanType, blockSize, Reduction, int >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned int >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned int >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, long > : public CudaBlockScanShfl< scanType, blockSize, Reduction, long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, long long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, long long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned long long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned long long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, float >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, float >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, double >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, double >
{};

/* Template for cooperative scan of a data tile in the global memory.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 */
template< ScanType scanType, int blockSize, int valuesPerThread, typename Reduction, typename ValueType >
struct CudaTileScan
{
   using BlockScan = CudaBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

   // storage to be allocated in shared memory
   using Storage = CudaTileScanStorage< ValueType, typename BlockScan::Storage, blockSize, valuesPerThread >;

   /* Cooperative scan of a data tile in the global memory - each thread will
    * get the result of its chunk (i.e. the last value of the (inclusive) scan
    * in the chunk) according to the thread ID.
    *
    * \param input        The input array to be scanned.
    * \param output       The array where the result will be stored.
    * \param begin        The first element in the array to be scanned.
    * \param end          the last element in the array to be scanned.
    * \param outputBegin  The first element in the output array to be written. There
    *                     must be at least `end - begin` elements in the output
    *                     array starting at the position given by `outputBegin`.
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param shift        A global shift to be applied to all elements in the
    *                     chunk processed by this thread.
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   template< typename InputView, typename OutputView >
   __device__
   static ValueType
   scan(
      const InputView input,
      OutputView output,
      typename InputView::IndexType begin,
      typename InputView::IndexType end,
      typename OutputView::IndexType outputBegin,
      const Reduction& reduction,
      ValueType identity,
      ValueType shift,
      Storage& storage )
   {
      // verify the configuration
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaTileScan::scan" );
      static_assert(
         valuesPerThread % 2,
         "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
         "when threads access their chunks in shared memory sequentially" );

      // calculate indices
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
      const int elementsInBlock = TNL::min( remainingElements, maxElementsInBlock );

      // update global array offsets for the thread
      const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
      begin += threadOffset;
      outputBegin += threadOffset;

      // Load data into the shared memory.
      {
         int idx = threadIdx.x;
         while( idx < elementsInBlock ) {
            storage.data[ idx ] = input[ begin ];
            begin += blockDim.x;
            idx += blockDim.x;
         }
         // fill the remaining (maxElementsInBlock - elementsInBlock) values with identity
         // (this helps to avoid divergent branches in the blocks below)
         while( idx < maxElementsInBlock ) {
            storage.data[ idx ] = identity;
            idx += blockDim.x;
         }
      }
      __syncthreads();

      // Perform sequential reduction of the thread's chunk in shared memory.
      const int chunkOffset = threadIdx.x * valuesPerThread;
      ValueType value = storage.data[ chunkOffset ];
   #pragma unroll
      for( int i = 1; i < valuesPerThread; i++ )
         value = reduction( value, storage.data[ chunkOffset + i ] );

      // Scan the spine to obtain the initial value ("offset") for the downsweep.
      value = BlockScan::scan( reduction, identity, value, threadIdx.x, storage.blockScanStorage );

      // Apply the global shift.
      value = reduction( value, shift );

   // Downsweep step: scan the chunks and use the result of spine scan as the initial value.
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

      // Store the result back in the global memory.
      {
         int idx = threadIdx.x;
         while( idx < elementsInBlock ) {
            output[ outputBegin ] = storage.data[ idx ];
            outputBegin += blockDim.x;
            idx += blockDim.x;
         }
      }

      // Return the last (inclusive) scan value of the chunk processed by this thread.
      return value;
   }
};
#endif

/* CudaScanKernelUpsweep - compute partial reductions per each CUDA block.
 */
template< int blockSize, int valuesPerThread, typename InputView, typename Reduction, typename ValueType >
__global__
void
CudaScanKernelUpsweep(
   const InputView input,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   Reduction reduction,
   ValueType identity,
   ValueType* reductionResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // verify the configuration
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelUpsweep" );
   static_assert(
      valuesPerThread % 2,
      "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
      "when threads access their chunks in shared memory sequentially" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduce< blockSize, Reduction, ValueType >;
   union Shared
   {
      ValueType data[ blockSize * valuesPerThread ];
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__
      Shared() {}
   };
   __shared__ Shared storage;

   // calculate indices
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
   const int elementsInBlock = TNL::min( remainingElements, maxElementsInBlock );

   // update global array offset for the thread
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   begin += threadOffset;

   // Load data into the shared memory.
   {
      int idx = threadIdx.x;
      while( idx < elementsInBlock ) {
         storage.data[ idx ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
      // fill the remaining (maxElementsInBlock - elementsInBlock) values with identity
      // (this helps to avoid divergent branches in the blocks below)
      while( idx < maxElementsInBlock ) {
         storage.data[ idx ] = identity;
         idx += blockDim.x;
      }
   }
   __syncthreads();

   // Perform sequential reduction of the thread's chunk in shared memory.
   const int chunkOffset = threadIdx.x * valuesPerThread;
   ValueType value = storage.data[ chunkOffset ];
   #pragma unroll
   for( int i = 1; i < valuesPerThread; i++ )
      value = reduction( value, storage.data[ chunkOffset + i ] );
   __syncthreads();

   // Perform the parallel reduction.
   value = BlockReduce::reduce( reduction, identity, value, storage.blockReduceStorage, threadIdx.x );

   // Store the block result in the global memory.
   if( threadIdx.x == 0 )
      reductionResults[ blockIdx.x ] = value;
#endif
}

/* CudaScanKernelDownsweep - scan each tile of the input separately in each CUDA
 * block and use the result of spine scan as the initial value
 */
template< ScanType scanType, int blockSize, int valuesPerThread, typename InputView, typename OutputView, typename Reduction >
__global__
void
CudaScanKernelDownsweep(
   const InputView input,
   OutputView output,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   typename OutputView::IndexType outputBegin,
   Reduction reduction,
   typename OutputView::ValueType identity,
   typename OutputView::ValueType shift,
   const typename OutputView::ValueType* reductionResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ValueType = typename OutputView::ValueType;
   using TileScan = CudaTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   __shared__ Backend::Uninitialized< typename TileScan::Storage > storage;

   // load the reduction of the previous tiles
   shift = reduction( shift, reductionResults[ blockIdx.x ] );

   // scan from input into output
   TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, shift, storage.get() );
#endif
}

/* CudaScanKernelParallel - scan each tile of the input separately in each CUDA
 * block (first phase to be followed by CudaScanKernelUniformShift when there
 * are multiple CUDA blocks).
 */
template< ScanType scanType, int blockSize, int valuesPerThread, typename InputView, typename OutputView, typename Reduction >
__global__
void
CudaScanKernelParallel(
   const InputView input,
   OutputView output,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   typename OutputView::IndexType outputBegin,
   Reduction reduction,
   typename OutputView::ValueType identity,
   typename OutputView::ValueType* blockResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ValueType = typename OutputView::ValueType;
   using TileScan = CudaTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   __shared__ Backend::Uninitialized< typename TileScan::Storage > storage;

   // scan from input into output
   const ValueType value =
      TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, identity, storage.get() );

   // The last thread of the block stores the block result in the global memory.
   if( blockResults && threadIdx.x == blockDim.x - 1 )
      blockResults[ blockIdx.x ] = value;
#endif
}

/* CudaScanKernelUniformShift - apply a uniform shift to a pre-scanned output
 * array.
 *
 * \param blockResults  An array of per-block shifts coming from the first phase
 *                      (computed by CudaScanKernelParallel)
 * \param shift         A global shift to be applied to all elements of the
 *                      output array.
 */
template< int blockSize, int valuesPerThread, typename OutputView, typename Reduction >
__global__
void
CudaScanKernelUniformShift(
   OutputView output,
   typename OutputView::IndexType outputBegin,
   typename OutputView::IndexType outputEnd,
   Reduction reduction,
   const typename OutputView::ValueType* blockResults,
   typename OutputView::ValueType shift )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // load the block result into a __shared__ variable first
   __shared__ Backend::Uninitialized< typename OutputView::ValueType > storage;
   if( threadIdx.x == 0 )
      storage.get() = blockResults[ blockIdx.x ];

   // update the output offset for the thread
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelUniformShift" );
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   outputBegin += threadOffset;

   // update the block shift
   __syncthreads();
   shift = reduction( shift, storage.get() );

   int valueIdx = 0;
   while( valueIdx < valuesPerThread && outputBegin < outputEnd ) {
      output[ outputBegin ] = reduction( output[ outputBegin ], shift );
      outputBegin += blockDim.x;
      valueIdx++;
   }
#endif
}

/* CudaScanKernelLookback - single-pass parallel prefix scan using decoupled
 * lookback synchronisation between CUDA blocks. Each block computes its local
 * scan and per-block aggregate, publishes the aggregate via the LookbackState
 * array, then walks predecessor aggregates to assemble its own prefix. Blocks
 * that finish their prefix early advertise it so later blocks can stop the
 * lookback walk sooner. No second kernel launch is needed.
 *
 * Reference: D. Merrill and M. Garland, "Single-pass Parallel Prefix Sum with
 * Decoupled Lookback", NVIDIA Research 2016.
 * https://research.nvidia.com/publication/2016-03_Single-pass-Parallel-Prefix-Sum-Decoupled-Lookback
 */
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
CudaScanKernelLookback(
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
   using TileScan = CudaTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;
   using BlockScan = CudaBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

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

/**
 * \brief Launcher for CUDA scan kernels.
 *
 * \tparam blockSize  The CUDA block size to be used for kernel launch.
 * \tparam valuesPerThread  Number of elements processed by each thread sequentially.
 */
template<
   ScanType scanType,
   ScanPhaseType phaseType,
   typename ValueType,
   // use blockSize=256 for 32-bit value types, scale with sizeof(ValueType)
   // to keep shared memory requirements constant
   int blockSize = 256 * 4 / sizeof( ValueType ),
   // valuesPerThread should be odd to avoid shared memory bank conflicts
   int valuesPerThread = 7 >
struct CudaScanKernelLauncher
{
   /**
    * \brief Performs both phases of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    */
   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   perform(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      if( end - begin <= blockSize * valuesPerThread ) {
         const auto blockShifts = performFirstPhase( input, output, begin, end, outputBegin, reduction, identity );
         return;
      }

      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
      const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      if( numberOfGrids == 1 ) {
         performLookback( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
         return;
      }

      const auto blockShifts = performFirstPhase( input, output, begin, end, outputBegin, reduction, identity );
      if( blockShifts.getSize() > 2 )
         performSecondPhase( input, output, blockShifts, begin, end, outputBegin, reduction, identity, identity );
   }

   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   performLookback(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename InputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      using Index = typename InputArray::IndexType;
      const Index n = end - begin;

      performLookbackImpl< valuesPerThread >(
         input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
   }

   template< int elementsPerThread, typename InputArray, typename OutputArray, typename Reduction >
   static void
   performLookbackImpl(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename InputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * elementsPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );

      // Reuse a thread-local cache of the lookback states array across calls
      // to avoid cudaMalloc/cudaFree churn. The cache grows as needed; when
      // the requested size shrinks we keep the existing (larger) allocation
      // and only memset the actually used prefix. Memset is still required
      // because the kernel reads predecessor statuses during the lookback
      // walk and they must start as Invalid.
      auto& states = lookbackStatesCache();
      if( states.getSize() < numberOfBlocks ) {
         states.setSize( numberOfBlocks );
      }
#if defined( __CUDACC__ )
      cudaMemsetAsync( states.getData(), 0, numberOfBlocks * sizeof( LookbackState< ValueType > ), 0 );
#elif defined( __HIP__ )
      hipMemsetAsync( states.getData(), 0, numberOfBlocks * sizeof( LookbackState< ValueType > ), 0 );
#endif

      constexpr auto kernel = CudaScanKernelLookback<
         scanType,
         blockSize,
         elementsPerThread,
         typename InputArray::ConstViewType,
         typename OutputArray::ViewType,
         std::decay_t< Reduction >,
         ValueType >;

      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = blockSize;
      launch_config.gridSize.x = numberOfBlocks;

      Backend::launchKernelSync(
         kernel,
         launch_config,
         input.getConstView(),
         output.getView(),
         begin,
         end,
         outputBegin,
         reduction,
         identity,
         states.getData() );

      gridsCount() = 1;
   }

   /**
    * \brief Performs the first phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    */
   template< typename InputArray, typename OutputArray, typename Reduction >
   static auto
   performFirstPhase(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      if( end - begin <= blockSize * valuesPerThread ) {
         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( 2 );
         blockResults.setElement( 0, identity );

         // run the kernel with just 1 block
         if( end - begin <= blockSize ) {
            constexpr auto kernel = CudaScanKernelParallel<
               scanType,
               blockSize,
               1,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else if( end - begin <= blockSize * 3 ) {
            constexpr auto kernel = CudaScanKernelParallel<
               scanType,
               blockSize,
               3,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else if( end - begin <= blockSize * 5 ) {
            constexpr auto kernel = CudaScanKernelParallel<
               scanType,
               blockSize,
               5,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else {
            constexpr auto kernel = CudaScanKernelParallel<
               scanType,
               blockSize,
               valuesPerThread,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = 1;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( numberOfBlocks + 1 );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // set CUDA launch configuration
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = blockSize;
            launch_config.gridSize.x = Backend::getNumberOfBlocks( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel = CudaScanKernelParallel<
                        scanType,
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        typename OutputArray::ViewType,
                        Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        input.getConstView(),
                        output.getView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        outputBegin + gridOffset,
                        reduction,
                        identity,
                        &blockResults.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }

               case ScanPhaseType::WriteInSecondPhase:
                  {
                     constexpr auto kernel = CudaScanKernelUpsweep<
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        Reduction,
                        typename OutputArray::ValueType >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        input.getConstView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        reduction,
                        identity,
                        &blockResults.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }
            }
         }

         // No sync needed: null-stream kernel launches are implicitly ordered.
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::perform(
            blockResults, blockResults, 0, blockResults.getSize(), 0, reduction, identity );

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = numberOfGrids;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
      }
   }

   /**
    * \brief Performs the second phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param blockShifts  Pointer to a GPU array containing the block shifts. It is the
    *                     result of the first phase.
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    * \param shift A constant shifting all elements of the array (usually
    *              `identity`, i.e. the neutral value).
    */
   template< typename InputArray, typename OutputArray, typename BlockShifts, typename Reduction >
   static void
   performSecondPhase(
      const InputArray& input,
      OutputArray& output,
      const BlockShifts& blockShifts,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      typename OutputArray::ValueType shift )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      // if the input was already scanned with just one block in the first phase,
      // it must be shifted uniformly in the second phase
      if( end - begin <= blockSize * valuesPerThread ) {
         constexpr auto kernel =
            CudaScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
         Backend::launchKernelSync(
            kernel,
            Backend::LaunchConfiguration( 1, blockSize ),
            output.getView(),
            outputBegin,
            outputBegin + end - begin,
            reduction,
            blockShifts.getData(),
            shift );
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // set CUDA launch configuration
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = blockSize;
            launch_config.gridSize.x = Backend::getNumberOfBlocks( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel =
                        CudaScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        output.getView(),
                        outputBegin + gridOffset,
                        outputBegin + gridOffset + currentSize,
                        reduction,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ],
                        shift );
                     break;
                  }

               case ScanPhaseType::WriteInSecondPhase:
                  {
                     constexpr auto kernel = CudaScanKernelDownsweep<
                        scanType,
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        typename OutputArray::ViewType,
                        Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        input.getConstView(),
                        output.getView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        outputBegin + gridOffset,
                        reduction,
                        identity,
                        shift,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }
            }
         }

         // synchronize the null-stream after all grids
         Backend::streamSynchronize( 0 );  // NOLINT(modernize-use-nullptr)
      }
   }

   // The following serves for setting smaller maxGridSize so that we can force
   // the scan in CUDA to run with more than one grid in unit tests.
   static std::size_t&
   maxGridSize()
   {
      static std::size_t maxGridSize = Backend::getMaxGridXSize();
      return maxGridSize;
   }

   static void
   resetMaxGridSize()
   {
      maxGridSize() = Backend::getMaxGridXSize();
      gridsCount() = -1;
   }

   // Per-instantiation cache of the lookback states array, reused across
   // calls to performLookbackImpl to avoid cudaMalloc/cudaFree churn.
   static Containers::Array< LookbackState< ValueType >, Devices::Cuda >&
   lookbackStatesCache()
   {
      static Containers::Array< LookbackState< ValueType >, Devices::Cuda > states;
      return states;
   }

   static int&
   gridsCount()
   {
      static int gridsCount = -1;
      return gridsCount;
   }
};

}  // namespace TNL::Algorithms::detail
