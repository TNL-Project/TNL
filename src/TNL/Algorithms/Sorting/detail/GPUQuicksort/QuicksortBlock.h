// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/Assert.h>
#include <TNL/Algorithms/Sorting/detail/BitonicSortBlock.h>
#include <TNL/Algorithms/detail/PivotPartitionBlock.h>
#include <TNL/Devices/GPU.h>

#include "../QuicksortPivot.h"

namespace TNL::Algorithms::Sorting::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Sentinel-aware fast bitonic sort of one block-local range.
 *
 * Specialized for the second phase of the GPU quicksort: it uses the classic
 * XOR-partner formulation of the bitonic network, which needs no per-element
 * bounds checks and no bank-conflicting index arithmetic, and pads the range
 * up to a power of two with a sentinel value that orders after every element.
 * The sentinel exists only for arithmetic value types sorted with std::less
 * (the sort order of the quicksort's public API); everything else falls back
 * to the generic \ref bitonicSortBlock.
 *
 * \param src view of the input range (must not overlap `dst`)
 * \param dst view of the output range, same size as `src`
 * \param sharedMem staging buffer with capacity for at least
 *        `closestPow2(src.getSize())` elements
 */
template< typename Value, typename Index, typename Compare >
__device__
void
quicksortBitonicSort(
   Containers::ArrayView< Value, TNL::Devices::GPU, Index > src,
   Containers::ArrayView< Value, TNL::Devices::GPU, Index > dst,
   Value* sharedMem,
   const Compare& compare )
{
   constexpr bool canPad = std::is_arithmetic_v< Value >
                        && ( std::is_same_v< std::decay_t< Compare >, std::less< void > >
                             || std::is_same_v< std::decay_t< Compare >, std::less< Value > > );
   const Index size = src.getSize();
   const unsigned logPaddedSize = closestPow2Log2( size );
   const Index paddedSize = Index( 1 ) << logPaddedSize;

   if constexpr( ! canPad ) {
      // no sentinel for the padding available, use the generic bounds-checked
      // network (it also handles non-power-of-two sizes without padding)
      bitonicSortBlock( src, dst, sharedMem, compare );
   }
   else {
      // the sentinel must not order before any element of the input, otherwise
      // it would move into the sorted range and evict real elements: for
      // floating-point types, `max()` orders before `+inf`, so use `infinity()`
      // (equal to `+inf` elements, which is harmless) and `max()` elsewhere
      constexpr Value sentinel = std::numeric_limits< Value >::has_infinity ? std::numeric_limits< Value >::infinity()
                                                                            : std::numeric_limits< Value >::max();
      for( Index i = threadIdx.x; i < size; i += blockDim.x )
         sharedMem[ i ] = src[ i ];
      for( Index i = size + threadIdx.x; i < paddedSize; i += blockDim.x )
         sharedMem[ i ] = sentinel;
      __syncthreads();

      for( unsigned logMonotonicSeqLen = 1; logMonotonicSeqLen <= logPaddedSize; ++logMonotonicSeqLen ) {
         const Index bitonicLen = Index( 1 ) << logMonotonicSeqLen;
         for( unsigned logStride = logMonotonicSeqLen; logStride-- > 0; ) {
            const Index stride = Index( 1 ) << logStride;
            // p enumerates the pairs: the lower element of pair p is
            // p + (p & -stride), the partner is the same element with the
            // stride bit set
            const Index pairCount = paddedSize / 2;
            for( Index p = threadIdx.x; p < pairCount; p += blockDim.x ) {
               const Index i = p + ( p & -stride );
               const Index partner = i | stride;
               const bool ascending = ( i & bitonicLen ) == 0;
               Value& lo = sharedMem[ i ];
               Value& hi = sharedMem[ partner ];
               if( ascending == compare( hi, lo ) )
                  swapValues( lo, hi );
            }
            __syncthreads();
         }
      }

      for( Index i = threadIdx.x; i < size; i += blockDim.x )
         dst[ i ] = sharedMem[ i ];
   }
}

/**
 * \brief Block-wide stack of the sub-ranges waiting to be partitioned.
 */
template< int stackSize, typename Index >
struct QuicksortBlockStack
{
   int top;
   Index begins[ stackSize ];
   Index ends[ stackSize ];
   int depths[ stackSize ];

   __device__
   void
   init( Index size, int iteration )
   {
      if( threadIdx.x == 0 ) {
         top = 0;
         begins[ 0 ] = 0;
         ends[ 0 ] = size;
         depths[ 0 ] = iteration;
         top++;
      }
      __syncthreads();
   }

   [[nodiscard]] __device__
   bool
   empty() const
   {
      return top <= 0;
   }

   __device__
   void
   pop( Index& begin, Index& end, int& iteration )
   {
      if( threadIdx.x == 0 ) {
         begin = begins[ top - 1 ];
         end = ends[ top - 1 ];
         iteration = depths[ top - 1 ];
         top--;
      }
   }

   __device__
   void
   push( Index begin, Index pivotBegin, Index pivotEnd, Index end, int iteration );
};

template< typename Value, typename Index, typename Compare, int stackSize, bool useShared >
__device__
void
quicksortBlock(
   Containers::ArrayView< Value, TNL::Devices::GPU, Index > arr,
   Containers::ArrayView< Value, TNL::Devices::GPU, Index > aux,
   const Compare& compare,
   int _iteration,
   Value* sharedMem,
   int memSize,
   int maxBitonicSize )
{
   if( arr.getSize() <= maxBitonicSize ) {
      auto& src = ( _iteration & 1 ) == 0 ? arr : aux;
      if( useShared && arr.getSize() <= memSize ) {
         quicksortBitonicSort( src, arr, sharedMem, compare );
      }
      else {
         bitonicSortBlock( src, compare );
         // the bitonic sort without shared memory works only in place, so copy from aux
         if( ( _iteration & 1 ) != 0 )
            for( Index i = threadIdx.x; i < arr.getSize(); i += blockDim.x )
               arr[ i ] = src[ i ];
      }

      return;
   }

   __shared__ QuicksortBlockStack< stackSize, Index > stack;
   static __shared__ Index begin;
   static __shared__ Index end;
   static __shared__ int iteration;
   static __shared__ Index pivotBegin;
   static __shared__ Index pivotEnd;
   Value* piv = sharedMem;
   sharedMem += 1;

   stack.init( arr.getSize(), _iteration );

   while( ! stack.empty() ) {
      // pick up the partition to break up
      stack.pop( begin, end, iteration );
      __syncthreads();

      Index size = end - begin;
      auto& src = ( iteration & 1 ) == 0 ? arr : aux;

      // small enough for bitonic sort
      if( size <= maxBitonicSize ) {
         if( useShared && size <= memSize ) {
            quicksortBitonicSort( src.getView( begin, end ), arr.getView( begin, end ), sharedMem, compare );
         }
         else {
            bitonicSortBlock( src.getView( begin, end ), compare );
            // the bitonic sort without shared memory works only in place, so copy from aux
            if( ( iteration & 1 ) != 0 )
               for( Index i = threadIdx.x; i < src.getSize(); i += blockDim.x )
                  arr[ begin + i ] = src[ i ];
         }
         __syncthreads();
         continue;
      }

      if( threadIdx.x == 0 )
         *piv = pickMedianOfThree( src.getView( begin, end ), compare ).first;
      __syncthreads();
      Value& pivot = *piv;

      const auto scan =
         Algorithms::detail::scanPivotPartitionBlock< Algorithms::detail::PivotPartitionEqualPolicy::FillLater, int >(
            src.getView( begin, end ), pivot, compare );

      if( threadIdx.x == blockDim.x - 1 )  // has sum of all smaller and greater elements than pivot in src
      {
         pivotBegin = static_cast< Index >( scan.total.less );
         pivotEnd = size - static_cast< Index >( scan.total.greater );
      }
      __syncthreads();

      // move elements, either use shared memory for coalesced access or without shared memory if data is too big

      auto& dst = ( iteration & 1 ) == 0 ? aux : arr;

      if( useShared && size <= memSize ) {
         Algorithms::detail::scatterPivotPartitionBlock< Algorithms::detail::PivotPartitionEqualPolicy::FillLater, true >(
            src.getView( begin, end ),
            dst.getView( begin, end ),
            pivot,
            compare,
            scan,
            0,
            0,  // equalOutputBegin is unused with FillLater
            pivotEnd,
            sharedMem );
      }
      else {
         Algorithms::detail::scatterPivotPartitionBlock< Algorithms::detail::PivotPartitionEqualPolicy::FillLater, false >(
            src.getView( begin, end ),
            dst.getView( begin, end ),
            pivot,
            compare,
            scan,
            0,
            0,  // equalOutputBegin is unused with FillLater
            pivotEnd,
            sharedMem );
      }

      __syncthreads();

      // arr is the canonical output buffer of this sorter and aux is only
      // scratch: the partition scatters into dst (the other buffer) and the
      // sub-partitions are resolved back into arr, so the equal middle must
      // be filled in arr, never in dst
      for( Index i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x )
         arr[ begin + i ] = pivot;

      // creates new tasks
      if( threadIdx.x == 0 )
         stack.push( begin, begin + pivotBegin, begin + pivotEnd, end, iteration );
      __syncthreads();  // sync to update the stack top
   }  // ends while loop
}

template< int stackSize, typename Index >
__device__
void
QuicksortBlockStack< stackSize, Index >::push( Index begin, Index pivotBegin, Index pivotEnd, Index end, int iteration )
{
   Index sizeL = pivotBegin - begin;
   Index sizeR = end - pivotEnd;

   // push the bigger one 1st and then smaller one 2nd
   // in next iteration, the smaller part will be handled 1st
   if( sizeL > sizeR ) {
      if( sizeL > 0 )  // left from pivot are smaller elements
      {
         begins[ top ] = begin;
         ends[ top ] = pivotBegin;
         depths[ top ] = iteration + 1;
         top++;
      }

      if( sizeR > 0 )  // right from pivot until end are elements greater than pivot
      {
         TNL_ASSERT_LT( top, stackSize, "Local quicksort stack overflow." );

         begins[ top ] = pivotEnd;
         ends[ top ] = end;
         depths[ top ] = iteration + 1;
         top++;
      }
   }
   else {
      if( sizeR > 0 )  // right from pivot until end are elements greater than pivot
      {
         begins[ top ] = pivotEnd;
         ends[ top ] = end;
         depths[ top ] = iteration + 1;
         top++;
      }

      if( sizeL > 0 )  // left from pivot are smaller elements
      {
         TNL_ASSERT_LT( top, stackSize, "Local quicksort stack overflow." );

         begins[ top ] = begin;
         ends[ top ] = pivotBegin;
         depths[ top ] = iteration + 1;
         top++;
      }
   }
}

#endif

}  // namespace TNL::Algorithms::Sorting::detail
