// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <stdexcept>

#include <TNL/Backend.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/detail/blockBitonicSort.h>
#include <TNL/Algorithms/Sorting/detail/closestPow2.h>

namespace TNL::Algorithms::Sorting::detail {

/**
 * \brief This kernel simulates 1 exchange.
 *
 * It splits input arr that is bitonic into 2 bitonic sequences.
 *
 * \param logMonotonicSeqLen Base-2 logarithm of the length of the monotonic sequences.
 * \param logBitonicLen Base-2 logarithm of the length of the bitonic sequences.
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicMergeGlobal(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr,
   CMP compare,
   unsigned int logMonotonicSeqLen,
   unsigned int logBitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Global thread index - must use Index type to avoid overflow for large arrays
   Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );

   // All lengths are powers of two, so division and multiplication are replaced
   // by shifts and masks - 64-bit integer division by a runtime value is very slow.
   const unsigned int logHalfBitonicLen = logBitonicLen - 1;
   const Index halfBitonicLen = Index( 1 ) << logHalfBitonicLen;

   // Compute which bitonic block this thread belongs to
   const Index part = i >> logHalfBitonicLen;

   // Calculate the two indices to be compared and swapped
   const Index s = ( part << logBitonicLen ) | ( i & ( halfBitonicLen - 1 ) );
   const Index e = s + halfBitonicLen;
   if( e >= arr.getSize() )  // arr[e] is virtual padding and will not be exchanged with
      return;

   // Determine the sorting direction (ascending/descending)
   const Index monotonicSeqIdx = i >> ( logMonotonicSeqLen - 1 );
   bool ascending = ( monotonicSeqIdx & 1 ) != 0;
   // Special case: last part has no "partner" to merge with in next phase
   if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= arr.getSize() )
      ascending = true;

   if( ascending == compare( arr[ e ], arr[ s ] ) )
      swapValues( arr[ s ], arr[ e ] );
#endif
}

//---------------------------------------------
//---------------------------------------------

/**
 * \brief Compares and swaps one pair of elements in the shared memory.
 *
 * The tile index `tl` enumerates the compared pairs, so the element indices are
 * `s = ( tl / halfLen ) * bitonicLen + tl % halfLen` and `e = s + halfLen`.
 *
 * \tparam uniformDirection Whether the sorting direction is the same for all pairs
 *                          in the tile, i.e. `blockAscending` can be used directly.
 */
template< bool uniformDirection, typename Value, typename Index, typename CMP >
__device__
void
bitonicMergePair(
   Value* sharedMem,
   Index tl,
   Index halfLen,
   unsigned int logBitonicLen,
   Index blockLen,
   bool blockAscending,
   Index firstPairIdx,
   unsigned int logMonotonicSeqLen,
   Index size,
   const CMP& compare )
{
   const Index p = tl >> ( logBitonicLen - 1 );
   const Index s = ( p << logBitonicLen ) | ( tl & ( halfLen - 1 ) );
   const Index e = s + halfLen;

   // Bounds check to avoid touching virtual padding
   if( e >= blockLen )
      return;

   bool ascending = blockAscending;
   if constexpr( ! uniformDirection ) {
      // The tile spans multiple sequences, so the direction varies per pair
      const Index monotonicSeqIdx = ( firstPairIdx + tl ) >> ( logMonotonicSeqLen - 1 );
      ascending = ( monotonicSeqIdx & 1 ) != 0;
      if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= size )
         ascending = true;
   }

   if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
      swapValues( sharedMem[ s ], sharedMem[ e ] );
}

/**
 * \brief Simulates many layers of merge.
 *
 * Turns input that is a bitonic sequence into 1 monotonic sequence.
 * This version uses shared memory to do the operations.
 *
 * \tparam uniformDirection Whether the whole tile belongs to a single monotonic
 *                          sequence, i.e. `tileLen` does not exceed the monotonic
 *                          sequence length. It allows to determine the sorting
 *                          direction just once per block.
 * \tparam singlePairPerThread Whether the tile has at most one compared pair per
 *                             thread, which avoids the loop over the pairs.
 * \param tileLen Number of elements processed by one block. It can span multiple
 *                monotonic sequences, but it must be a power of two that is not
 *                smaller than the bitonic length of the first merge phase.
 * \param logMonotonicSeqLen Base-2 logarithm of the length of the monotonic sequences.
 * \param logBitonicLen Base-2 logarithm of the length of the first bitonic sequence to merge.
 */
template< bool uniformDirection, bool singlePairPerThread, typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicMergeSharedMemory(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr,
   CMP compare,
   Index tileLen,
   unsigned int logMonotonicSeqLen,
   unsigned int logBitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = reinterpret_cast< Value* >( externMem );

   // Range of elements this block processes
   const Index size = static_cast< Index >( arr.getSize() );
   const Index myBlockStart = static_cast< Index >( blockIdx.x ) * tileLen;
   const Index blockLen = TNL::min( size, myBlockStart + tileLen ) - myBlockStart;
   const Index numPairs = tileLen / 2;
   // Global index of the first compared pair in this block
   const Index firstPairIdx = myBlockStart >> 1;

   // Determine the sorting direction of the monotonic sequence containing this tile
   bool blockAscending = false;
   if constexpr( uniformDirection ) {
      const Index monotonicSeqIdx = firstPairIdx >> ( logMonotonicSeqLen - 1 );
      blockAscending = ( monotonicSeqIdx & 1 ) != 0;
      // Special case: last part has no "partner"
      if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= size )
         blockAscending = true;
   }

   // Copy from global memory to shared memory
   for( Index i = threadIdx.x; i < blockLen; i += blockDim.x )
      sharedMem[ i ] = arr[ myBlockStart + i ];
   __syncthreads();

   //------------------------------------------
   // Bitonic merge phase
   {
      // Perform bitonic merge in shared memory
      for( ; logBitonicLen > 0; --logBitonicLen ) {
         const Index halfLen = Index( 1 ) << ( logBitonicLen - 1 );
         if constexpr( singlePairPerThread ) {
            bitonicMergePair< uniformDirection >(
               sharedMem,
               Index( threadIdx.x ),
               halfLen,
               logBitonicLen,
               blockLen,
               blockAscending,
               firstPairIdx,
               logMonotonicSeqLen,
               size,
               compare );
         }
         else {
            for( Index tl = threadIdx.x; tl < numPairs; tl += blockDim.x )
               bitonicMergePair< uniformDirection >(
                  sharedMem,
                  tl,
                  halfLen,
                  logBitonicLen,
                  blockLen,
                  blockAscending,
                  firstPairIdx,
                  logMonotonicSeqLen,
                  size,
                  compare );
         }
         __syncthreads();
      }
   }

   //------------------------------------------

   // Write back to global memory
   for( Index i = threadIdx.x; i < blockLen; i += blockDim.x )
      arr[ myBlockStart + i ] = sharedMem[ i ];
#endif
}

/**
 * \brief Entrypoint for bitonicSort_Block.
 *
 * Sorts `arr` in alternating order to create bitonic sequences.
 * `sharedMem` has to be able to store at least `blockDim.x * 2` elements.
 *
 * \param arr The array to sort in alternating order.
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicSortFirstStepSharedMemory( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr, CMP compare )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];

   Value* sharedMem = reinterpret_cast< Value* >( externMem );
   int sharedMemLen = 2 * blockDim.x;

   // Range of elements this block processes
   Index myBlockStart = blockIdx.x * sharedMemLen;
   Index myBlockEnd = TNL::min( arr.getSize(), myBlockStart + sharedMemLen );

   // Copy from global memory to shared memory
   for( Index i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x )
      sharedMem[ i ] = arr[ myBlockStart + i ];
   __syncthreads();

   //------------------------------------------
   // Bitonic sort phase - creates alternating sorted sequences
   {
      // Global thread index - must use Index type to avoid overflow for large arrays
      Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );
      const unsigned int logPaddedSize = closestPow2Log2( myBlockEnd - myBlockStart );

      for( unsigned int logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
         // Determine sorting direction for this sequence
         Index monotonicSeqIdx = i >> ( logMonotonicSeqLen - 1 );
         bool ascending = ( monotonicSeqIdx & 1 ) != 0;
         // Special case: last part has no "partner"
         if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= arr.getSize() )
            ascending = true;

         // Perform bitonic merge
         for( unsigned int logLen = logMonotonicSeqLen; logLen > 0; --logLen ) {
            // Calculate which 2 indexes will be compared and swapped
            Index halfLen = Index( 1 ) << ( logLen - 1 );
            Index p = threadIdx.x >> ( logLen - 1 );
            Index s = ( p << logLen ) | ( threadIdx.x & ( halfLen - 1 ) );
            Index e = s + halfLen;

            // Bounds check to avoid touching virtual padding
            if( e < myBlockEnd - myBlockStart )
               if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
                  swapValues( sharedMem[ s ], sharedMem[ e ] );
            __syncthreads();
         }
      }
   }

   // Write back to global memory
   for( Index i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x )
      arr[ myBlockStart + i ] = sharedMem[ i ];
#endif
}

/**
 * \brief Bitonic sort using shared memory for the merge phases that fit into it.
 *
 * The first step sorts tiles of `2 * blockDim` elements in shared memory, which
 * makes the array a sequence of alternating monotonic sequences. The merge phases
 * then process the longest bitonic lengths that still fit into the shared memory
 * of one block. Only the phases with larger bitonic lengths have to go through
 * global memory, which makes them the dominant part of the running time.
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
void
bitonicSortWithShared(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view,
   const CMP& compare,
   int blockDim,
   Index firstStepTileLen,
   Index mergeTileLen )
{
   const Index size = view.getSize();
   const unsigned int logPaddedSize = closestPow2Log2( size );
   const unsigned int logFirstStepTileLen = closestPow2Log2( firstStepTileLen );
   const unsigned int logMergeTileLen = closestPow2Log2( mergeTileLen );

   // First step: sort tiles in shared memory to create alternating monotonic sequences
   Backend::LaunchConfiguration first_step_config;
   first_step_config.blockSize.x = blockDim;
   first_step_config.gridSize.x = roundUpDivision( size, firstStepTileLen );
   first_step_config.dynamicSharedMemorySize = firstStepTileLen * sizeof( Value );
   {
      constexpr auto kernel = bitonicSortFirstStepSharedMemory< Value, ArrayIndex, CMP, Index >;
      Backend::launchKernelAsync( kernel, first_step_config, view, compare );
   }

   // Configuration for the merge phases that are too large for the shared memory
   Backend::LaunchConfiguration global_config;
   global_config.blockSize.x = blockDim;
   global_config.gridSize.x = roundUpDivision( size, Index( 2 * blockDim ) );

   // Configuration for the merge phases processed in shared memory
   Backend::LaunchConfiguration shared_config;
   shared_config.blockSize.x = blockDim;
   shared_config.gridSize.x = roundUpDivision( size, mergeTileLen );
   shared_config.dynamicSharedMemorySize = mergeTileLen * sizeof( Value );

   // Merge phases: combine bitonic sequences into larger sorted sequences
   for( unsigned int logMonotonicSeqLen = logFirstStepTileLen + 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
      // Merges with a bitonic length larger than the tile have to use global memory
      for( unsigned int logBitonicLen = logMonotonicSeqLen; logBitonicLen > logMergeTileLen; --logBitonicLen ) {
         constexpr auto kernel = bitonicMergeGlobal< Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, global_config, view, compare, logMonotonicSeqLen, logBitonicLen );
      }

      // The shared memory kernel processes all remaining bitonic lengths internally
      const unsigned int logBitonicLen = TNL::min( logMonotonicSeqLen, logMergeTileLen );
      const bool uniformDirection = logMonotonicSeqLen >= logMergeTileLen;
      const bool singlePairPerThread = mergeTileLen == Index( 2 * blockDim );
      if( uniformDirection && singlePairPerThread ) {
         constexpr auto kernel = bitonicMergeSharedMemory< true, true, Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, shared_config, view, compare, mergeTileLen, logMonotonicSeqLen, logBitonicLen );
      }
      else if( uniformDirection ) {
         constexpr auto kernel = bitonicMergeSharedMemory< true, false, Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, shared_config, view, compare, mergeTileLen, logMonotonicSeqLen, logBitonicLen );
      }
      else if( singlePairPerThread ) {
         constexpr auto kernel = bitonicMergeSharedMemory< false, true, Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, shared_config, view, compare, mergeTileLen, logMonotonicSeqLen, logBitonicLen );
      }
      else {
         constexpr auto kernel = bitonicMergeSharedMemory< false, false, Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, shared_config, view, compare, mergeTileLen, logMonotonicSeqLen, logBitonicLen );
      }
   }
   Backend::streamSynchronize( first_step_config.stream );
}

//---------------------------------------------

/**
 * \brief Bitonic sort using only global memory (no shared memory optimization).
 *
 * Used when shared memory is too small for the element type.
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
void
bitonicSort( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view, const CMP& compare, Index gridDim, int blockDim )

{
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = blockDim;
   launch_config.gridSize.x = gridDim;

   const unsigned int logPaddedSize = closestPow2Log2( view.getSize() );

   // Iterate through all merge phases
   for( unsigned int logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
      for( unsigned int logBitonicLen = logMonotonicSeqLen; logBitonicLen > 0; --logBitonicLen ) {
         constexpr auto kernel = bitonicMergeGlobal< Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, launch_config, view, compare, logMonotonicSeqLen, logBitonicLen );
      }
   }
   Backend::streamSynchronize( launch_config.stream );
}

//---------------------------------------------

/**
 * \brief Main entry point for bitonic sort on CUDA arrays.
 *
 * Automatically selects between shared memory and global memory implementations
 * based on available shared memory and element size.
 */
template< typename Value, typename ArrayIndex, typename CMP >
void
bitonicSort( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view, const CMP& compare )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // Early exit for empty or single-element arrays (already sorted)
   if( view.getSize() <= 1 )
      return;

   // Ensure that at least 32-bit type is used for indexing
   using Index = std::common_type_t< ArrayIndex, unsigned int >;

   // The kernels shift by the logarithms of the sequence lengths, so the padded
   // size has to be representable in the index type
   if( closestPow2Log2( view.getSize() ) >= static_cast< unsigned int >( std::numeric_limits< Index >::digits ) )
      throw std::length_error( "bitonic sort: the array size exceeds the range of the index type" );

   const std::size_t sharedMemPerBlock = Backend::getSharedMemoryPerBlock( Backend::getDevice() );

   // Reduce the block size if the first step does not fit into the shared memory
   int blockDim = 512;
   while( blockDim > 64 && static_cast< std::size_t >( 2 * blockDim ) * sizeof( Value ) > sharedMemPerBlock )
      blockDim /= 2;

   // TODO: handle integer overflow - launch multiple grids?
   if( static_cast< std::size_t >( 2 * blockDim ) * sizeof( Value ) <= sharedMemPerBlock ) {
      const Index firstStepTileLen = 2 * blockDim;

      // Use as many elements per block as the shared memory allows - every merge phase
      // that fits into the shared memory saves two passes through the global memory
      std::size_t mergeTileLen = firstStepTileLen;
      while( 2 * mergeTileLen * sizeof( Value ) <= sharedMemPerBlock )
         mergeTileLen *= 2;

      // Small arrays do not have enough tiles to keep the GPU busy, so shrink the tile
      const std::size_t minBlocks = 8 * static_cast< std::size_t >( Backend::getDeviceMultiprocessors( Backend::getDevice() ) );
      while( mergeTileLen > static_cast< std::size_t >( firstStepTileLen )
             && mergeTileLen * minBlocks > static_cast< std::size_t >( view.getSize() ) )
         mergeTileLen /= 2;

      bitonicSortWithShared( view, compare, blockDim, firstStepTileLen, static_cast< Index >( mergeTileLen ) );
   }
   else {
      // Use global memory only when shared memory is too small. The global kernel
      // does not use the shared memory, so it can keep the full block size.
      constexpr int globalBlockDim = 512;
      const Index gridDim = roundUpDivision( Index( view.getSize() ), Index( 2 * globalBlockDim ) );
      bitonicSort( view, compare, gridDim, globalBlockDim );
   }
#endif
}

//---------------------------------------------
//---------------------------------------------

/**
 * \brief Bitonic merge kernel for fetch-and-swap interface.
 *
 * Used when elements are accessed via custom fetch/swap functions rather than
 * direct array access.
 */
template< typename Index, typename CMP, typename SWAP >
__global__
void
bitonicMergeGlobalWithSwap( Index size, CMP compare, SWAP Swap, unsigned int logMonotonicSeqLen, unsigned int logBitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Global thread index - must use Index type to avoid overflow for large arrays
   Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );

   const unsigned int logHalfBitonicLen = logBitonicLen - 1;
   const Index halfBitonicLen = Index( 1 ) << logHalfBitonicLen;

   // Compute which bitonic block this thread belongs to
   const Index part = i >> logHalfBitonicLen;

   // Calculate the two indices to be compared and swapped
   const Index s = ( part << logBitonicLen ) | ( i & ( halfBitonicLen - 1 ) );
   const Index e = s + halfBitonicLen;
   if( e >= size )  // e is virtual padding - skip
      return;

   // Determine sorting direction
   const Index monotonicSeqIdx = i >> ( logMonotonicSeqLen - 1 );
   bool ascending = ( monotonicSeqIdx & 1 ) != 0;
   // Special case: last part has no "partner"
   if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= size )
      ascending = true;

   if( ascending == compare( e, s ) )
      Swap( s, e );
#endif
}

/**
 * \brief Bitonic sort with custom fetch and swap operations.
 *
 * Useful for sorting indirect arrays or when additional logic is needed
 * during comparison or swapping.
 */
template< typename Index, typename CMP, typename SWAP >
void
bitonicSort( Index begin, Index end, const CMP& compare, SWAP Swap )
{
   Index size = end - begin;

   // Early exit for empty or single-element ranges (already sorted)
   if( size <= 1 )
      return;

   // The kernels shift by the logarithms of the sequence lengths, so the padded
   // size has to be representable in the type used for indexing inside the kernels.
   // The kernels use at least 32-bit indices even when the range is given by a
   // narrower index type, so the check must use the widened type.
   using ComputeIndex = std::common_type_t< Index, unsigned int >;
   if( closestPow2Log2( size ) >= static_cast< unsigned int >( std::numeric_limits< ComputeIndex >::digits ) )
      throw std::length_error( "bitonic sort: the range exceeds the range of the index type" );

   const unsigned int logPaddedSize = closestPow2Log2( size );

   Index threadsNeeded = roundUpDivision( size, Index{ 2 } );

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = 512;
   // TODO: handle integer overflow - launch multiple grids?
   launch_config.gridSize.x = Backend::getNumberOfBlocks( threadsNeeded, launch_config.blockSize.x );

   // Wrap compare and swap functions to handle offset
   auto compareWithOffset = [ = ] __cuda_callable__( Index i, Index j )
   {
      return compare( i + begin, j + begin );
   };

   auto swapWithOffset = [ = ] __cuda_callable__( Index i, Index j ) mutable
   {
      Swap( i + begin, j + begin );
   };

   // Iterate through all merge phases
   for( unsigned int logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
      for( unsigned int logBitonicLen = logMonotonicSeqLen; logBitonicLen > 0; --logBitonicLen ) {
         constexpr auto kernel =
            bitonicMergeGlobalWithSwap< ComputeIndex, decltype( compareWithOffset ), decltype( swapWithOffset ) >;
         Backend::launchKernelAsync(
            kernel, launch_config, size, compareWithOffset, swapWithOffset, logMonotonicSeqLen, logBitonicLen );
      }
   }
   Backend::streamSynchronize( launch_config.stream );
}

}  // namespace TNL::Algorithms::Sorting::detail
