// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/detail/closestPow2.h>
#include <TNL/Containers/ArrayView.h>

namespace TNL::Algorithms::Sorting::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src and writes into dst within a block
 * works independently from other concurrent blocks
 * @param sharedMem sharedMem pointer has to be able to store all of src elements
 * */
template< typename Value, typename Index, typename CMP >
__device__
void
bitonicSort_Block(
   Containers::ArrayView< Value, Devices::GPU, Index > src,
   Containers::ArrayView< Value, Devices::GPU, Index > dst,
   Value* sharedMem,
   const CMP& compare )
{
   // Copy from global memory to shared memory
   for( Index i = threadIdx.x; i < src.getSize(); i += blockDim.x )
      sharedMem[ i ] = src[ i ];
   __syncthreads();

   //------------------------------------------
   // Bitonic sort phase
   {
      const Index paddedSize = closestPow2_ptx( src.getSize() );

      for( Index monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
         for( Index bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
            // Iterate over all pairs in this bitonic merge step
            // Loop handles cases where src.size > blockDim.x*2 by simulating multiple blocks
            for( Index i = threadIdx.x;; i += blockDim.x ) {
               Index halfBitonicLen = bitonicLen / 2;
               Index p = i / halfBitonicLen;
               Index s = p * bitonicLen + ( i & ( halfBitonicLen - 1 ) );
               Index e = s + halfBitonicLen;

               // Bounds check - avoid touching virtual padding
               if( e >= src.getSize() )
                  break;

               // Determine sorting direction
               Index halfMonotonicSeqLen = monotonicSeqLen / 2;
               Index monotonicSeqIdx = i / halfMonotonicSeqLen;
               bool ascending = ( monotonicSeqIdx & 1 ) != 0;
               // Special case: last part has no "partner"
               if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= src.getSize() )
                  ascending = true;

               if( ascending == compare( sharedMem[ e ], sharedMem[ s ] ) )
                  TNL::swap( sharedMem[ s ], sharedMem[ e ] );
            }

            __syncthreads();
         }
      }
   }

   // Write back to destination
   for( Index i = threadIdx.x; i < dst.getSize(); i += blockDim.x )
      dst[ i ] = sharedMem[ i ];
}

/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * IMPORTANT: unlike the counterpart with shared memory, this function only works in-place
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src in place using bitonic sort
 * works independently from other concurrent blocks
 * this version doesnt use shared memory and is preferred for Value with big size
 * */
template< typename Value, typename Index, typename CMP >
__device__
void
bitonicSort_Block( Containers::ArrayView< Value, Devices::GPU, Index > src, const CMP& compare )
{
   const Index paddedSize = closestPow2_ptx( src.getSize() );

   for( Index monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
      for( Index bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
         // Iterate over all pairs in this bitonic merge step
         // Loop handles cases where src.size > blockDim.x*2 by simulating multiple blocks
         for( Index i = threadIdx.x;; i += blockDim.x ) {
            Index halfBitonicLen = bitonicLen / 2;
            Index p = i / halfBitonicLen;
            Index s = p * bitonicLen + ( i & ( halfBitonicLen - 1 ) );
            Index e = s + halfBitonicLen;

            // Bounds check - avoid touching virtual padding
            if( e >= src.getSize() )
               break;

            // Determine sorting direction
            Index halfMonotonicSeqLen = monotonicSeqLen / 2;
            Index monotonicSeqIdx = i / halfMonotonicSeqLen;
            bool ascending = ( monotonicSeqIdx & 1 ) != 0;
            // Special case: last part has no "partner"
            if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= src.getSize() )
               ascending = true;

            if( ascending == compare( src[ e ], src[ s ] ) )
               TNL::swap( src[ s ], src[ e ] );
         }
         __syncthreads();
      }
   }
}

#endif

}  // namespace TNL::Algorithms::Sorting::detail
