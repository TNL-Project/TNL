// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Cooperative scan of a tile of blockSize * valuesPerThread elements by one
 * CUDA block.
 *
 * The tile is loaded into shared memory, each thread reduces its chunk of
 * valuesPerThread consecutive elements sequentially, the chunk results are
 * scanned across the block by GPUBlockScan, and finally each thread scans its
 * chunk again with the block-wide prefix as the initial value and the result
 * is stored in the global memory.
 *
 * It is used by GPUScanKernelParallel and GPUScanKernelDownsweep;
 * GPUScanKernelLookback uses only its shared memory storage.
 */

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>

namespace TNL::Algorithms::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename ValueType, typename BlockStorage, int BlockSize, int ValuesPerThread >
struct GPUTileScanStorage
{
   alignas( 16 ) ValueType data[ BlockSize * ValuesPerThread ];
   BlockStorage blockScanStorage;
};

/* Template for cooperative scan of a data tile in the global memory.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 */
template< ScanType scanType, int blockSize, int valuesPerThread, typename Reduction, typename ValueType >
struct GPUTileScan
{
   using BlockScan = GPUBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

   // storage to be allocated in shared memory
   using Storage = GPUTileScanStorage< ValueType, typename BlockScan::Storage, blockSize, valuesPerThread >;

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
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in GPUTileScan::scan" );
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

}  // namespace TNL::Algorithms::detail
