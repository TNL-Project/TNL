// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Sorting/detail/closestPow2.h>
#include <TNL/Containers/ArrayView.h>

namespace TNL::Algorithms::Sorting::detail {

/**
 * \brief Swaps two values in host and device code.
 *
 * `std::swap` is not used because NVCC treats it as host-only
 * for value types without `constexpr` special member functions.
 *
 * \param a The first value.
 * \param b The second value.
 */
template< typename Value >
__cuda_callable__
constexpr void
swapValues( Value& a, Value& b ) noexcept
{
   Value tmp = a;
   a = b;
   b = tmp;
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Sorts an array by one block of threads using the shared memory.
 *
 * The elements of `src` are loaded into `sharedMem`, sorted there,
 * and the first `dst.getSize()` sorted elements are written into `dst`.
 * The arrays `src` and `dst` may be the same array.
 * The block works independently of the other blocks.
 * The array size is not limited,
 * but the function is meant for short arrays,
 * because all the work is done by one block.
 *
 * \warning All threads of the block must call this function,
 * because it synchronizes them with `__syncthreads()`.
 * The threads are synchronized before the function returns,
 * so `dst` can be read and `sharedMem` can be reused by any thread of the block.
 *
 * \tparam Value Type of the array elements.
 * \tparam Index Type of the array indices.
 * \tparam CMP Type of the comparison function.
 *
 * \param src The array to be sorted.
 * \param dst The array for the result, not longer than `src`.
 * \param sharedMem Pointer to the shared memory which can hold `src.getSize()` elements.
 * \param compare Strict weak ordering of the elements.
 */
template< typename Value, typename Index, typename CMP >
__device__
void
bitonicSortBlock(
   Containers::ArrayView< Value, Devices::GPU, Index > src,
   Containers::ArrayView< Value, Devices::GPU, Index > dst,
   Value* sharedMem,
   const CMP& compare )
{
   // Use at least 32-bit indices, the shifts below would overflow narrow types
   using ComputeIndex = std::common_type_t< Index, unsigned int >;
   const ComputeIndex size = src.getSize();

   // Copy from global memory to shared memory
   for( ComputeIndex i = threadIdx.x; i < size; i += blockDim.x )
      sharedMem[ i ] = src[ i ];
   __syncthreads();

   // Bitonic sort phase
   {
      const unsigned int logPaddedSize = closestPow2Log2( size );

      for( unsigned int logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
         for( unsigned int logBitonicLen = logMonotonicSeqLen; logBitonicLen > 0; --logBitonicLen ) {
            const ComputeIndex halfBitonicLen = ComputeIndex( 1 ) << ( logBitonicLen - 1 );
            // Iterate over all pairs in this bitonic merge step
            // Loop handles cases where src.size > blockDim.x*2 by simulating multiple blocks
            for( ComputeIndex i = threadIdx.x;; i += blockDim.x ) {
               ComputeIndex p = i >> ( logBitonicLen - 1 );
               ComputeIndex s = ( p << logBitonicLen ) | ( i & ( halfBitonicLen - 1 ) );
               ComputeIndex e = s + halfBitonicLen;

               // Bounds check - avoid touching virtual padding
               if( e >= size )
                  break;

               // Determine sorting direction
               ComputeIndex monotonicSeqIdx = i >> ( logMonotonicSeqLen - 1 );
               bool ascending = ( monotonicSeqIdx & 1 ) != 0;
               // Special case: last part has no "partner"
               if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= size )
                  ascending = true;

               if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
                  swapValues( sharedMem[ s ], sharedMem[ e ] );
            }

            __syncthreads();
         }
      }
   }

   // Write back to destination
   for( ComputeIndex i = threadIdx.x; i < static_cast< ComputeIndex >( dst.getSize() ); i += blockDim.x )
      dst[ i ] = sharedMem[ i ];
   // Make the result visible to the whole block and allow reusing the shared memory
   __syncthreads();
}

/**
 * \brief Sorts an array in place by one block of threads without the shared memory.
 *
 * The compare-and-swap operations access the elements of `src` directly,
 * so this variant is suitable for value types
 * which are too large for the shared memory.
 * The block works independently of the other blocks.
 * The array size is not limited,
 * but the function is meant for short arrays,
 * because all the work is done by one block.
 *
 * \warning All threads of the block must call this function,
 * because it synchronizes them with `__syncthreads()`.
 * The threads are synchronized before the function returns.
 *
 * \tparam Value Type of the array elements.
 * \tparam Index Type of the array indices.
 * \tparam CMP Type of the comparison function.
 *
 * \param src The array to be sorted in place.
 * \param compare Strict weak ordering of the elements.
 */
template< typename Value, typename Index, typename CMP >
__device__
void
bitonicSortBlock( Containers::ArrayView< Value, Devices::GPU, Index > src, const CMP& compare )
{
   // Use at least 32-bit indices, the shifts below would overflow narrow types
   using ComputeIndex = std::common_type_t< Index, unsigned int >;
   const ComputeIndex size = src.getSize();
   const unsigned int logPaddedSize = closestPow2Log2( size );

   for( unsigned int logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
      for( unsigned int logBitonicLen = logMonotonicSeqLen; logBitonicLen > 0; --logBitonicLen ) {
         const ComputeIndex halfBitonicLen = ComputeIndex( 1 ) << ( logBitonicLen - 1 );
         // Iterate over all pairs in this bitonic merge step
         // Loop handles cases where src.size > blockDim.x*2 by simulating multiple blocks
         for( ComputeIndex i = threadIdx.x;; i += blockDim.x ) {
            ComputeIndex p = i >> ( logBitonicLen - 1 );
            ComputeIndex s = ( p << logBitonicLen ) | ( i & ( halfBitonicLen - 1 ) );
            ComputeIndex e = s + halfBitonicLen;

            // Bounds check - avoid touching virtual padding
            if( e >= size )
               break;

            // Determine sorting direction
            ComputeIndex monotonicSeqIdx = i >> ( logMonotonicSeqLen - 1 );
            bool ascending = ( monotonicSeqIdx & 1 ) != 0;
            // Special case: last part has no "partner"
            if( ( monotonicSeqIdx + 1 ) << logMonotonicSeqLen >= size )
               ascending = true;

            if( ascending == compare( src[ e ], src[ s ] ) )
               swapValues( src[ s ], src[ e ] );
         }
         __syncthreads();
      }
   }
}

#endif

}  // namespace TNL::Algorithms::Sorting::detail
