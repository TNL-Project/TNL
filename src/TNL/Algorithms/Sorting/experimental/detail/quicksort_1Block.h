// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Assert.h>
#include <TNL/Algorithms/Sorting/detail/blockBitonicSort.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>

namespace TNL::Algorithms::Sorting::experimental::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Value, typename Index, typename Compare >
__device__
void
externSort(
   Containers::ArrayView< Value, TNL::Devices::Cuda, Index > src,
   Containers::ArrayView< Value, TNL::Devices::Cuda, Index > dst,
   const Compare& compare,
   Value* sharedMem )
{
   Sorting::detail::bitonicSort_Block( src, dst, sharedMem, compare );
}

template< typename Value, typename Index, typename Compare >
__device__
void
externSort( Containers::ArrayView< Value, TNL::Devices::Cuda, Index > src, const Compare& compare )
{
   Sorting::detail::bitonicSort_Block( src, compare );
}

template< int stackSize, typename Index >
__device__
void
stackPush(
   Index stackArrBegin[],
   Index stackArrEnd[],
   int stackDepth[],
   int& stackTop,
   Index begin,
   Index pivotBegin,
   Index pivotEnd,
   Index end,
   int iteration );

template< typename Value, typename Index, typename Compare, int stackSize, bool useShared >
__device__
void
singleBlockQuickSort(
   Containers::ArrayView< Value, TNL::Devices::Cuda, Index > arr,
   Containers::ArrayView< Value, TNL::Devices::Cuda, Index > aux,
   const Compare& compare,
   int _iteration,
   Value* sharedMem,
   int memSize,
   int maxBitonicSize )
{
   if( arr.getSize() <= maxBitonicSize ) {
      auto& src = ( _iteration & 1 ) == 0 ? arr : aux;
      if( useShared && arr.getSize() <= memSize )
         externSort< Value, Index, Compare >( src, arr, compare, sharedMem );
      else {
         externSort< Value, Index, Compare >( src, compare );
         // extern sort without shared memory only works in-place, need to copy into from aux
         if( ( _iteration & 1 ) != 0 )
            for( Index i = threadIdx.x; i < arr.getSize(); i += blockDim.x )
               arr[ i ] = src[ i ];
      }

      return;
   }

   static __shared__ int stackTop;
   static __shared__ Index stackArrBegin[ stackSize ];
   static __shared__ Index stackArrEnd[ stackSize ];
   static __shared__ int stackDepth[ stackSize ];
   static __shared__ Index begin;
   static __shared__ Index end;
   static __shared__ int iteration;
   static __shared__ Index pivotBegin;
   static __shared__ Index pivotEnd;
   Value* piv = sharedMem;
   sharedMem += 1;

   if( threadIdx.x == 0 ) {
      stackTop = 0;
      stackArrBegin[ stackTop ] = 0;
      stackArrEnd[ stackTop ] = arr.getSize();
      stackDepth[ stackTop ] = _iteration;
      stackTop++;
   }
   __syncthreads();

   while( stackTop > 0 ) {
      // pick up partition to break up
      if( threadIdx.x == 0 ) {
         begin = stackArrBegin[ stackTop - 1 ];
         end = stackArrEnd[ stackTop - 1 ];
         iteration = stackDepth[ stackTop - 1 ];
         stackTop--;
      }
      __syncthreads();

      Index size = end - begin;
      auto& src = ( iteration & 1 ) == 0 ? arr : aux;

      // small enough for bitonic sort
      if( size <= maxBitonicSize ) {
         if( useShared && size <= memSize )
            externSort< Value, Index, Compare >( src.getView( begin, end ), arr.getView( begin, end ), compare, sharedMem );
         else {
            externSort< Value, Index, Compare >( src.getView( begin, end ), compare );
            // extern sort without shared memory only works in-place, need to copy into from aux
            if( ( iteration & 1 ) != 0 )
               for( Index i = threadIdx.x; i < src.getSize(); i += blockDim.x )
                  arr[ begin + i ] = src[ i ];
         }
         __syncthreads();
         continue;
      }

      if( threadIdx.x == 0 )
         *piv = pickPivot( src.getView( begin, end ), compare );
      __syncthreads();
      Value& pivot = *piv;

      int smaller = 0;
      int bigger = 0;
      countElements( src.getView( begin, end ), compare, smaller, bigger, pivot );

      // synchronization is in this function already
      using BlockScan = Algorithms::detail::CudaBlockScan< Algorithms::detail::ScanType::Inclusive, 0, TNL::Plus, int >;
      __shared__ typename BlockScan::Storage storage;
      int smallerPrefSumInc = BlockScan::scan( TNL::Plus{}, 0, smaller, threadIdx.x, storage );
      int biggerPrefSumInc = BlockScan::scan( TNL::Plus{}, 0, bigger, threadIdx.x, storage );

      if( threadIdx.x == blockDim.x - 1 )  // has sum of all smaller and greater elements than pivot in src
      {
         pivotBegin = static_cast< Index >( smallerPrefSumInc );
         pivotEnd = size - static_cast< Index >( biggerPrefSumInc );
      }
      __syncthreads();

      /**
       * move elements, either use shared memory for coalesced access or without shared memory if data is too big
       * */

      auto& dst = ( iteration & 1 ) == 0 ? aux : arr;

      if( useShared && size <= memSize ) {
         static __shared__ int smallerTotal;
         static __shared__ int biggerTotal;
         if( threadIdx.x == blockDim.x - 1 ) {
            smallerTotal = smallerPrefSumInc;
            biggerTotal = biggerPrefSumInc;
         }
         __syncthreads();

         copyDataShared(
            src.getView( begin, end ),
            dst.getView( begin, end ),
            compare,
            sharedMem,
            static_cast< Index >( 0 ),
            pivotEnd,
            smallerTotal,
            biggerTotal,
            smallerPrefSumInc - smaller,
            biggerPrefSumInc - bigger,  // exclusive prefix sum of elements
            pivot );
      }
      else {
         Index destSmaller = static_cast< Index >( smallerPrefSumInc - smaller );
         Index destBigger = pivotEnd + static_cast< Index >( biggerPrefSumInc - bigger );

         copyData( src.getView( begin, end ), dst.getView( begin, end ), compare, destSmaller, destBigger, pivot );
      }

      __syncthreads();

      for( Index i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x )
         arr[ begin + i ] = pivot;

      // creates new tasks
      if( threadIdx.x == 0 ) {
         stackPush< stackSize, Index >(
            stackArrBegin, stackArrEnd, stackDepth, stackTop, begin, begin + pivotBegin, begin + pivotEnd, end, iteration );
      }
      __syncthreads();  // sync to update stackTop
   }  // ends while loop
}

template< int stackSize, typename Index >
__device__
void
stackPush(
   Index stackArrBegin[],
   Index stackArrEnd[],
   int stackDepth[],
   int& stackTop,
   Index begin,
   Index pivotBegin,
   Index pivotEnd,
   Index end,
   int iteration )
{
   Index sizeL = pivotBegin - begin;
   Index sizeR = end - pivotEnd;

   // push the bigger one 1st and then smaller one 2nd
   // in next iteration, the smaller part will be handled 1st
   if( sizeL > sizeR ) {
      if( sizeL > 0 )  // left from pivot are smaller elements
      {
         stackArrBegin[ stackTop ] = begin;
         stackArrEnd[ stackTop ] = pivotBegin;
         stackDepth[ stackTop ] = iteration + 1;
         stackTop++;
      }

      if( sizeR > 0 )  // right from pivot until end are elements greater than pivot
      {
         TNL_ASSERT_LT( stackTop, stackSize, "Local quicksort stack overflow." );

         stackArrBegin[ stackTop ] = pivotEnd;
         stackArrEnd[ stackTop ] = end;
         stackDepth[ stackTop ] = iteration + 1;
         stackTop++;
      }
   }
   else {
      if( sizeR > 0 )  // right from pivot until end are elements greater than pivot
      {
         stackArrBegin[ stackTop ] = pivotEnd;
         stackArrEnd[ stackTop ] = end;
         stackDepth[ stackTop ] = iteration + 1;
         stackTop++;
      }

      if( sizeL > 0 )  // left from pivot are smaller elements
      {
         TNL_ASSERT_LT( stackTop, stackSize, "Local quicksort stack overflow." );

         stackArrBegin[ stackTop ] = begin;
         stackArrEnd[ stackTop ] = pivotBegin;
         stackDepth[ stackTop ] = iteration + 1;
         stackTop++;
      }
   }
}

#endif

}  // namespace TNL::Algorithms::Sorting::experimental::detail
