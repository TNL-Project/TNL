// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/detail/blockBitonicSort.h>
#include <TNL/Algorithms/Sorting/detail/closestPow2.h>

namespace TNL::Algorithms::Sorting::detail {

/**
 * this kernel simulates 1 exchange
 * splits input arr that is bitonic into 2 bitonic sequences
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicMergeGlobal(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr,
   CMP compare,
   Index monotonicSeqLen,
   Index bitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Global thread index - must use Index type to avoid overflow for large arrays
   Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );

   // Compute which bitonic block this thread belongs to
   Index halfBitonicLen = bitonicLen / 2;
   Index part = i / halfBitonicLen;

   // Calculate the two indices to be compared and swapped
   Index s = part * bitonicLen + ( i & ( halfBitonicLen - 1 ) );
   Index e = s + halfBitonicLen;
   if( e >= arr.getSize() )  // arr[e] is virtual padding and will not be exchanged with
      return;

   // Determine the sorting direction (ascending/descending)
   Index partsInSeq = monotonicSeqLen / bitonicLen;
   Index monotonicSeqIdx = part / partsInSeq;
   bool ascending = ( monotonicSeqIdx & 1 ) != 0;
   // Special case: last part has no "partner" to merge with in next phase
   if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= arr.getSize() )
      ascending = true;

   if( ascending == compare( arr[ e ], arr[ s ] ) )
      TNL::swap( arr[ s ], arr[ e ] );
#endif
}

//---------------------------------------------
//---------------------------------------------

/**
 * simulates many layers of merge
 * turns input that is a bitonic sequence into 1 monotonic sequence
 *
 * this version uses shared memory to do the operations
 * */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicMergeSharedMemory(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr,
   CMP compare,
   Index monotonicSeqLen,
   Index bitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = (Value*) externMem;

   int sharedMemLen = 2 * blockDim.x;

   // Range of elements this block processes
   Index myBlockStart = blockIdx.x * sharedMemLen;
   Index myBlockEnd = TNL::min( arr.getSize(), myBlockStart + sharedMemLen );

   // Copy from global memory to shared memory
   for( Index i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x )
      sharedMem[ i ] = arr[ myBlockStart + i ];
   __syncthreads();

   //------------------------------------------
   // Bitonic merge phase
   {
      // Global thread index - must use Index type to avoid overflow for large arrays
      Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );
      Index halfBitonicLen = bitonicLen / 2;
      Index part = i / halfBitonicLen;
      Index partsInSeq = monotonicSeqLen / bitonicLen;
      Index monotonicSeqIdx = part / partsInSeq;

      // Determine sorting direction
      bool ascending = ( monotonicSeqIdx & 1 ) != 0;
      // Special case: last part has no "partner"
      if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= static_cast< Index >( arr.getSize() ) )
         ascending = true;
      //------------------------------------------

      // Perform bitonic merge in shared memory
      for( ; bitonicLen > 1; bitonicLen /= 2 ) {
         // Calculate which 2 indexes will be compared and swapped
         Index halfLen = bitonicLen / 2;
         Index p = threadIdx.x / halfLen;
         Index s = p * bitonicLen + ( threadIdx.x & ( halfLen - 1 ) );
         Index e = s + halfLen;

         // Bounds check to avoid touching virtual padding
         if( e < myBlockEnd - myBlockStart )
            if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
               TNL::swap( sharedMem[ s ], sharedMem[ e ] );
         __syncthreads();
      }
   }

   //------------------------------------------

   // Write back to global memory
   for( Index i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x )
      arr[ myBlockStart + i ] = sharedMem[ i ];
#endif
}

/**
 * entrypoint for bitonicSort_Block
 * sorts @param arr in alternating order to create bitonic sequences
 * sharedMem has to be able to store at least blockDim.x*2 elements
 * */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
__global__
void
bitonicSortFirstStepSharedMemory( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > arr, CMP compare )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];

   Value* sharedMem = (Value*) externMem;
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
      Index paddedSize = closestPow2_ptx( myBlockEnd - myBlockStart );

      for( Index monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
         // Determine sorting direction for this sequence
         Index halfMonotonicSeqLen = monotonicSeqLen / 2;
         Index monotonicSeqIdx = i / halfMonotonicSeqLen;
         bool ascending = ( monotonicSeqIdx & 1 ) != 0;
         // Special case: last part has no "partner"
         if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= arr.getSize() )
            ascending = true;

         // Perform bitonic merge
         for( Index len = monotonicSeqLen; len > 1; len /= 2 ) {
            // Calculate which 2 indexes will be compared and swapped
            Index halfLen = len / 2;
            Index p = threadIdx.x / halfLen;
            Index s = p * len + ( threadIdx.x & ( halfLen - 1 ) );
            Index e = s + halfLen;

            // Bounds check to avoid touching virtual padding
            if( e < myBlockEnd - myBlockStart )
               if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
                  TNL::swap( sharedMem[ s ], sharedMem[ e ] );
            __syncthreads();
         }
      }
   }

   // Write back to global memory
   for( Index i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x )
      arr[ myBlockStart + i ] = sharedMem[ i ];
#endif
}

template< typename Value, typename ArrayIndex, typename CMP, typename Index >
void
bitonicSortWithShared(
   Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view,
   const CMP& compare,
   Index gridDim,
   int blockDim,
   int sharedMemLen,
   std::size_t sharedMemSize )
{
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = blockDim;
   launch_config.gridSize.x = gridDim;
   launch_config.dynamicSharedMemorySize = sharedMemSize;

   const Index paddedSize = closestPow2( static_cast< Index >( view.getSize() ) );
   const int sharedMemLenIndex = sharedMemLen;

   // First step: sort blocks in shared memory to create alternating monotonic sequences
   constexpr auto kernel = bitonicSortFirstStepSharedMemory< Value, ArrayIndex, CMP, Index >;
   Backend::launchKernelAsync( kernel, launch_config, view, compare );
   // Now we have alternating monotonic sequences with bitonicLength of sharedMemLen

   // Merge phases: combine bitonic sequences into larger sorted sequences
   // Starts with bitonicLength of 2 * sharedMemLen
   for( Index monotonicSeqLen = 2 * sharedMemLenIndex; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
      for( Index bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
         if( bitonicLen > static_cast< Index >( sharedMemLenIndex ) ) {
            // Use global memory kernel for large bitonic lengths
            launch_config.dynamicSharedMemorySize = 0;
            constexpr auto kernel = bitonicMergeGlobal< Value, ArrayIndex, CMP, Index >;
            Backend::launchKernelAsync( kernel, launch_config, view, compare, monotonicSeqLen, bitonicLen );
         }
         else {
            // Use shared memory kernel for small bitonic lengths (faster)
            launch_config.dynamicSharedMemorySize = sharedMemSize;
            constexpr auto kernel = bitonicMergeSharedMemory< Value, ArrayIndex, CMP, Index >;
            Backend::launchKernelAsync( kernel, launch_config, view, compare, monotonicSeqLen, bitonicLen );

            // Shared memory kernel processes all remaining bitonic lengths internally
            break;
         }
      }
   }
   Backend::streamSynchronize( launch_config.stream );
}

//---------------------------------------------

/**
 * Bitonic sort using only global memory (no shared memory optimization).
 * Used when shared memory is too small for the element type.
 */
template< typename Value, typename ArrayIndex, typename CMP, typename Index >
void
bitonicSort( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view, const CMP& compare, Index gridDim, int blockDim )

{
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = blockDim;
   launch_config.gridSize.x = gridDim;

   const Index paddedSize = closestPow2( static_cast< Index >( view.getSize() ) );

   // Iterate through all merge phases
   for( Index monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
      for( Index bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
         constexpr auto kernel = bitonicMergeGlobal< Value, ArrayIndex, CMP, Index >;
         Backend::launchKernelAsync( kernel, launch_config, view, compare, monotonicSeqLen, bitonicLen );
      }
   }
   Backend::streamSynchronize( launch_config.stream );
}

//---------------------------------------------

/**
 * Main entry point for bitonic sort on CUDA arrays.
 * Automatically selects between shared memory and global memory implementations
 * based on available shared memory and element size.
 */
template< typename Value, typename ArrayIndex, typename CMP >
void
bitonicSort( Containers::ArrayView< Value, Devices::GPU, ArrayIndex > view, const CMP& compare )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // Ensure that at least 32-bit type is used for indexing
   using Index = std::common_type_t< ArrayIndex, unsigned int >;

   const Index threadsNeeded = roundUpDivision( view.getSize(), ArrayIndex{ 2 } );
   constexpr int maxThreadsPerBlock = 512;
   int sharedMemLen = maxThreadsPerBlock * 2;
   std::size_t sharedMemSize = sharedMemLen * sizeof( Value );
   const std::size_t sharedMemPerBlock = Backend::getSharedMemoryPerBlock( Backend::getDevice() );

   // Try to use shared memory for better performance
   if( sharedMemSize <= sharedMemPerBlock ) {
      int blockDim = maxThreadsPerBlock;
      // TODO: handle integer overflow - launch multiple grids?
      Index gridDim = Backend::getNumberOfBlocks( threadsNeeded, blockDim );
      bitonicSortWithShared( view, compare, gridDim, blockDim, sharedMemLen, sharedMemSize );
   }
   else if( sharedMemSize / 2 <= sharedMemPerBlock ) {
      // Fall back to smaller block size if full shared memory does not fit
      int blockDim = maxThreadsPerBlock / 2;
      // TODO: handle integer overflow - launch multiple grids?
      Index gridDim = Backend::getNumberOfBlocks( threadsNeeded, blockDim );
      sharedMemSize /= 2;
      sharedMemLen /= 2;
      bitonicSortWithShared( view, compare, gridDim, blockDim, sharedMemLen, sharedMemSize );
   }
   else {
      // Use global memory only when shared memory is too small
      // TODO: handle integer overflow - launch multiple grids?
      Index gridDim = Backend::getNumberOfBlocks( threadsNeeded, maxThreadsPerBlock );
      bitonicSort( view, compare, gridDim, maxThreadsPerBlock );
   }
#endif
}

//---------------------------------------------
//---------------------------------------------

/**
 * Bitonic merge kernel for fetch-and-swap interface.
 * Used when elements are accessed via custom fetch/swap functions rather than
 * direct array access.
 */
template< typename Index, typename CMP, typename SWAP >
__global__
void
bitonicMergeGlobalWithSwap( Index size, CMP compare, SWAP Swap, Index monotonicSeqLen, Index bitonicLen )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Global thread index - must use Index type to avoid overflow for large arrays
   Index i = static_cast< Index >( blockIdx.x ) * static_cast< Index >( blockDim.x ) + static_cast< Index >( threadIdx.x );

   // Compute which bitonic block this thread belongs to
   Index halfBitonicLen = bitonicLen / 2;
   Index part = i / halfBitonicLen;

   // Calculate the two indices to be compared and swapped
   Index s = part * bitonicLen + ( i & ( halfBitonicLen - 1 ) );
   Index e = s + halfBitonicLen;
   if( e >= size )  // e is virtual padding - skip
      return;

   // Determine sorting direction
   Index partsInSeq = monotonicSeqLen / bitonicLen;
   Index monotonicSeqIdx = part / partsInSeq;
   bool ascending = ( monotonicSeqIdx & 1 ) != 0;
   // Special case: last part has no "partner"
   if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= size )
      ascending = true;

   if( ascending == compare( e, s ) )
      Swap( s, e );
#endif
}

/**
 * Bitonic sort with custom fetch and swap operations.
 * Useful for sorting indirect arrays or when additional logic is needed
 * during comparison or swapping.
 */
template< typename Index, typename CMP, typename SWAP >
void
bitonicSort( Index begin, Index end, const CMP& compare, SWAP Swap )
{
   Index size = end - begin;
   Index paddedSize = closestPow2( size );

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
   for( Index monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
      for( Index bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
         // Ensure that at least 32-bit type is used for indexing
         using ComputeIndex = std::common_type_t< Index, unsigned int >;
         constexpr auto kernel =
            bitonicMergeGlobalWithSwap< ComputeIndex, decltype( compareWithOffset ), decltype( swapWithOffset ) >;
         Backend::launchKernelAsync(
            kernel, launch_config, size, compareWithOffset, swapWithOffset, monotonicSeqLen, bitonicLen );
      }
   }
   Backend::streamSynchronize( launch_config.stream );
}

}  // namespace TNL::Algorithms::Sorting::detail
