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
 * The tile loads and stores are vectorized (16-byte accesses) for arrays
 * whose value type size divides 16.
 *
 * GPUTileScan::scan is used by GPUScanKernelParallel and
 * GPUScanKernelDownsweep; GPUScanKernelLookback uses the individual steps
 * (loadTile, reduceChunk, scanChunk, storeTile), since it determines the
 * prefix of the tile between them.
 */

#include <cstdint>  // std::uintptr_t

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>

namespace TNL::Algorithms::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

// 16-byte vector type used for vectorized loads/stores of the tiles. int4 is
// available on all CUDA targets and supports all value types whose sizeof
// divides 16 (1, 2, 4, 8, 16 bytes).
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
   static_assert(
      valuesPerThread % 2,
      "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
      "when threads access their chunks in shared memory sequentially" );

   using BlockScan = GPUBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

   // storage to be allocated in shared memory
   using Storage = GPUTileScanStorage< ValueType, typename BlockScan::Storage, blockSize, valuesPerThread >;

   static constexpr int maxElementsInTile = blockSize * valuesPerThread;

   /* Cooperative scan of a data tile in the global memory - each thread will
    * get the result of its chunk (i.e. the last value of the (inclusive) scan
    * in the chunk) according to the thread ID. The tile is selected by
    * blockIdx.x.
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

      // the offset is computed in the index type, since one grid may cover
      // more elements than int can represent
      using InputIndex = typename InputView::IndexType;
      const InputIndex tileOffset = static_cast< InputIndex >( blockIdx.x ) * maxElementsInTile;
      const int elementsInTile =
         static_cast< int >( TNL::min( end - begin - tileOffset, static_cast< InputIndex >( maxElementsInTile ) ) );

      loadTile( input, begin + tileOffset, elementsInTile, identity, storage );
      __syncthreads();

      ValueType value = reduceChunk( reduction, storage );

      // Scan the spine to obtain the initial value ("offset") for the downsweep.
      value = BlockScan::scan( reduction, identity, value, threadIdx.x, storage.blockScanStorage );

      // Apply the global shift.
      value = reduction( value, shift );

      value = scanChunk( reduction, value, storage );
      __syncthreads();

      storeTile( output, outputBegin + tileOffset, elementsInTile, storage );

      // Return the last (inclusive) scan value of the chunk processed by this thread.
      return value;
   }

   /* Loads elementsInTile elements starting at tileBegin into the shared
    * memory and pads the rest of the tile with identity, so that all threads
    * run the same code path in the following steps.
    *
    * The aligned middle of the tile is loaded via Vec4 for arrays whose value
    * type size divides 16 (with scalar shared stores, since the middle does
    * not start at a 16-byte boundary in shared memory in general); the
    * unaligned prefix and suffix are loaded scalarly. Expression templates are
    * loaded scalarly (no getData()). The front peel is uniform across the
    * block, so there is no warp divergence.
    */
   template< typename InputView >
   __device__
   static void
   loadTile(
      const InputView& input,
      typename InputView::IndexType tileBegin,
      int elementsInTile,
      ValueType identity,
      Storage& storage )
   {
      // range of the tile accessed via Vec4
      int vecBegin = elementsInTile;
      int vecEnd = elementsInTile;
      if constexpr( 16 % sizeof( ValueType ) == 0 && IsArrayType< InputView >::value ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         vecBegin = getVec4FrontPeel( input.getData() + tileBegin, elementsInTile );
         vecEnd = vecBegin + ( elementsInTile - vecBegin ) / vecWidth * vecWidth;
         const Vec4* inputVec = reinterpret_cast< const Vec4* >( input.getData() + tileBegin + vecBegin );
         for( int v = threadIdx.x; v < ( vecEnd - vecBegin ) / vecWidth; v += blockDim.x ) {
            const Vec4 vec = inputVec[ v ];
            const ValueType* values = reinterpret_cast< const ValueType* >( &vec );
   #pragma unroll
            for( int c = 0; c < vecWidth; c++ )
               storage.data[ vecBegin + v * vecWidth + c ] = values[ c ];
         }
      }
      for( int i = threadIdx.x; i < vecBegin; i += blockDim.x )
         storage.data[ i ] = input[ tileBegin + i ];
      for( int i = vecEnd + threadIdx.x; i < elementsInTile; i += blockDim.x )
         storage.data[ i ] = input[ tileBegin + i ];
      for( int i = elementsInTile + threadIdx.x; i < maxElementsInTile; i += blockDim.x )
         storage.data[ i ] = identity;
   }

   /* Stores the first elementsInTile elements of the tile from the shared
    * memory to the output starting at tileBegin. The accesses are vectorized
    * the same way as in loadTile.
    */
   template< typename OutputView >
   __device__
   static void
   storeTile( OutputView& output, typename OutputView::IndexType tileBegin, int elementsInTile, const Storage& storage )
   {
      // range of the tile accessed via Vec4
      int vecBegin = elementsInTile;
      int vecEnd = elementsInTile;
      if constexpr( 16 % sizeof( ValueType ) == 0 && IsArrayType< OutputView >::value ) {
         constexpr int vecWidth = 16 / sizeof( ValueType );
         vecBegin = getVec4FrontPeel( output.getData() + tileBegin, elementsInTile );
         vecEnd = vecBegin + ( elementsInTile - vecBegin ) / vecWidth * vecWidth;
         Vec4* outputVec = reinterpret_cast< Vec4* >( output.getData() + tileBegin + vecBegin );
         for( int v = threadIdx.x; v < ( vecEnd - vecBegin ) / vecWidth; v += blockDim.x ) {
            Vec4 vec;
            ValueType* values = reinterpret_cast< ValueType* >( &vec );
   #pragma unroll
            for( int c = 0; c < vecWidth; c++ )
               values[ c ] = storage.data[ vecBegin + v * vecWidth + c ];
            outputVec[ v ] = vec;
         }
      }
      for( int i = threadIdx.x; i < vecBegin; i += blockDim.x )
         output[ tileBegin + i ] = storage.data[ i ];
      for( int i = vecEnd + threadIdx.x; i < elementsInTile; i += blockDim.x )
         output[ tileBegin + i ] = storage.data[ i ];
   }

   // Returns the sequential reduction of the chunk of the calling thread in
   // the shared memory.
   __device__
   static ValueType
   reduceChunk( const Reduction& reduction, const Storage& storage )
   {
      const int chunkOffset = threadIdx.x * valuesPerThread;
      ValueType value = storage.data[ chunkOffset ];
   #pragma unroll
      for( int i = 1; i < valuesPerThread; i++ )
         value = reduction( value, storage.data[ chunkOffset + i ] );
      return value;
   }

   // Scans the chunk of the calling thread in the shared memory with the
   // given initial value (the exclusive prefix of the chunk) and returns the
   // last value of the inclusive scan of the chunk.
   __device__
   static ValueType
   scanChunk( const Reduction& reduction, ValueType value, Storage& storage )
   {
      const int chunkOffset = threadIdx.x * valuesPerThread;
   #pragma unroll
      for( int i = 0; i < valuesPerThread; i++ ) {
         const ValueType inputValue = storage.data[ chunkOffset + i ];
         if( scanType == ScanType::Exclusive )
            storage.data[ chunkOffset + i ] = value;
         value = reduction( value, inputValue );
         if( scanType == ScanType::Inclusive )
            storage.data[ chunkOffset + i ] = value;
      }
      return value;
   }
};

#endif

}  // namespace TNL::Algorithms::detail
