// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Containers/Array.h>
#include "QuicksortTask.h"
#include "cudaPartition.h"
#include "quicksort_1Block.h"

namespace TNL::Algorithms::Sorting::experimental::detail {

template< typename Index >
__device__
void
writeNewTask(
   Index begin,
   Index end,
   int iteration,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt );

//-----------------------------------------------------------

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Returns the offset of the sample with the given index.
 *
 * The samples are evenly spaced over the range. The quotient/remainder
 * decomposition keeps the intermediate products below the range size, so it does
 * not overflow for any index type.
 */
template< typename Index >
__device__
Index
sampleOffset( Index taskSize, int nSamples, int sample )
{
   const Index last = taskSize - 1;
   const Index divisor = static_cast< Index >( nSamples - 1 );
   return sample * ( last / divisor ) + sample * ( last % divisor ) / divisor;
}

/**
 * \brief Returns the index of the sample whose rank is closest to the middle.
 *
 * \param partitionBegin first index of the partitioned range
 * \param taskSize number of elements in the partitioned range
 * \param samples one sample per thread, stored in shared memory
 *
 * Must be called by the whole block. Ranks are computed by counting, so no
 * sorting network and only two barriers are needed. The returned index refers to
 * a real element of the range, which keeps the sort a permutation.
 */
template< typename Value, typename Index, typename Compare >
__device__
Index
pickSampledPivotIdx( Index partitionBegin, Index taskSize, int nSamples, const Value* samples, const Compare& compare )
{
   const int half = nSamples / 2;
   int rank = 0;
   if( threadIdx.x < nSamples ) {
      const Value mySample = samples[ threadIdx.x ];
      for( int j = 0; j < nSamples; ++j )
         rank += compare( samples[ j ], mySample ) ? 1 : 0;
   }
   else {
      rank = 2 * half;  // largest possible deviation, the thread can never win
   }

   const int deviation = rank > half ? rank - half : half - rank;
   // the deviation is packed with the thread index so that ties are broken
   // deterministically; a reduction is used instead of an atomic because
   // hundreds of threads serialized on one shared memory location cost more than
   // the rest of the kernel
   int packed = ( deviation << 16 ) | static_cast< int >( threadIdx.x );
   for( int offset = warpSize / 2; offset > 0; offset >>= 1 ) {
      const int other = Backend::warp_shuffle_down( packed, offset );
      packed = other < packed ? other : packed;
   }

   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   const int warp = static_cast< int >( threadIdx.x ) / warpSize;
   const int warps = static_cast< int >( blockDim.x ) / warpSize;
   __shared__ int warpBest[ 32 ];
   if( lane == 0 )
      warpBest[ warp ] = packed;
   __syncthreads();

   __shared__ Index pivotIdx;
   if( threadIdx.x == 0 ) {
      int best = warpBest[ 0 ];
      for( int w = 1; w < warps; ++w )
         best = warpBest[ w ] < best ? warpBest[ w ] : best;
      const int winner = best & 0xffff;
      pivotIdx = partitionBegin + sampleOffset( taskSize, nSamples, winner );
   }
   __syncthreads();
   return pivotIdx;
}

#endif

template< typename Value, typename Index, typename Compare >
__global__
void
cudaInitTask(
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > cuda_tasks,
   Containers::ArrayView< int, Devices::Cuda > cuda_blockToTaskMapping,
   Containers::ArrayView< int, Devices::Cuda > cuda_reductionTaskInitMem,
   Containers::ArrayView< Value, Devices::Cuda, Index > src,
   Compare compare,
   int nSamples,
   bool canBufferSamples )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( blockIdx.x >= cuda_tasks.getSize() )
      return;

   int start = blockIdx.x == 0 ? 0 : cuda_reductionTaskInitMem[ blockIdx.x - 1 ];
   int end = cuda_reductionTaskInitMem[ blockIdx.x ];
   for( int i = start + threadIdx.x; i < end; i += blockDim.x )
      cuda_blockToTaskMapping[ i ] = blockIdx.x;

   QuicksortTask< Index >& task = cuda_tasks[ blockIdx.x ];
   const Index taskSize = task.partitionEnd - task.partitionBegin;

   /*
    * The median of three elements gives splits around 70/30 on average, which
    * roughly doubles the number of partitioning rounds compared to the median of
    * the data. Sampling the range is much cheaper than a full median estimate and
    * the quality saturates quickly: the gain over the median of three is already
    * there for a few dozen samples, while every sample is a separate memory
    * transaction that is paid again for each task in every round.
    */
   // named differently than the buffers in the other kernels, because all extern
   // __shared__ declarations of the same name in a translation unit must be
   // compatible, but this one is aligned for arbitrary Value types
   extern __shared__ __align__( 16 ) unsigned char sampleMem[];
   Value* samples = reinterpret_cast< Value* >( sampleMem );
   const bool useSampledPivot = canBufferSamples && taskSize > static_cast< Index >( nSamples ) && nSamples > 1;
   Index pivotIdx;
   if( useSampledPivot ) {
      if( threadIdx.x < nSamples ) {
         const Index offset = sampleOffset( taskSize, nSamples, static_cast< int >( threadIdx.x ) );
         samples[ threadIdx.x ] = src[ task.partitionBegin + offset ];
      }
      __syncthreads();
      pivotIdx = pickSampledPivotIdx( task.partitionBegin, taskSize, nSamples, samples, compare );
   }
   else {
      pivotIdx = task.partitionBegin + pickPivot( src.getView( task.partitionBegin, task.partitionEnd ), compare ).second;
   }

   if( threadIdx.x == 0 )
      task.init( start, end - start, pivotIdx );
#endif
}

template< typename Value, typename Index, typename Compare, bool useShared >
__global__
void
cudaQuickSortFirstPhase(
   Containers::ArrayView< Value, Devices::Cuda, Index > arr,
   Containers::ArrayView< Value, Devices::Cuda, Index > aux,
   const Compare& compare,
   int elementsPerBlock,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > tasks,
   Containers::ArrayView< int, Devices::Cuda > taskMapping )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = reinterpret_cast< Value* >( externMem );
   Value* sharedMem = piv + 1;

   QuicksortTask< Index >& myTask = tasks[ taskMapping[ blockIdx.x ] ];
   auto& src = ( myTask.iteration & 1 ) == 0 ? arr : aux;
   auto& dst = ( myTask.iteration & 1 ) == 0 ? aux : arr;

   if( threadIdx.x == 0 )
      *piv = src[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   cudaPartition< Value, Index, Compare, useShared >(
      src.getView( myTask.partitionBegin, myTask.partitionEnd ),
      dst.getView( myTask.partitionBegin, myTask.partitionEnd ),
      compare,
      sharedMem,
      pivot,
      elementsPerBlock,
      myTask );
#endif
}

template< typename Value, typename Index >
__global__
void
cudaWritePivot(
   Containers::ArrayView< Value, Devices::Cuda, Index > arr,
   Containers::ArrayView< Value, Devices::Cuda, Index > aux,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > tasks,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = reinterpret_cast< Value* >( externMem );

   QuicksortTask< Index >& myTask = tasks[ blockIdx.x ];

   if( threadIdx.x == 0 )
      *piv = ( myTask.iteration & 1 ) == 0 ? arr[ myTask.pivotIdx ] : aux[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   Index leftBegin = myTask.partitionBegin;
   Index leftEnd = myTask.partitionBegin + static_cast< Index >( myTask.dstBegin );
   Index rightBegin = myTask.partitionBegin + static_cast< Index >( myTask.dstEnd );
   Index rightEnd = myTask.partitionEnd;

   for( Index i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x )
      arr[ i ] = pivot;

   if( threadIdx.x != 0 )
      return;

   if( leftEnd - leftBegin > 0 ) {
      writeNewTask(
         leftBegin,
         leftEnd,
         myTask.iteration,
         maxElementsForSecondPhase,
         newTasks,
         newTasksCnt,
         secondPhaseTasks,
         secondPhaseTasksCnt );
   }

   if( rightEnd - rightBegin > 0 ) {
      writeNewTask(
         rightBegin,
         rightEnd,
         myTask.iteration,
         maxElementsForSecondPhase,
         newTasks,
         newTasksCnt,
         secondPhaseTasks,
         secondPhaseTasksCnt );
   }
#endif
}

template< typename Index >
__device__
void
writeNewTask(
   Index begin,
   Index end,
   int iteration,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Index size = end - begin;
   TNL_ASSERT_GE( size, 0, "negative size in writeNewTask" );

   if( size == 0 )
      return;

   if( size <= maxElementsForSecondPhase ) {
      int idx = atomicAdd( secondPhaseTasksCnt, 1 );
      if( idx < secondPhaseTasks.getSize() ) {
         secondPhaseTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
      else {
         int idx = atomicAdd( newTasksCnt, 1 );
         TNL_ASSERT_LT( idx, newTasks.getSize(), "task memory exhausted in writeNewTask" );
         newTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
   }
   else {
      int idx = atomicAdd( newTasksCnt, 1 );
      if( idx < newTasks.getSize() ) {
         newTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
      else {
         int idx = atomicAdd( secondPhaseTasksCnt, 1 );
         TNL_ASSERT_LT( idx, secondPhaseTasks.getSize(), "task memory exhausted in writeNewTask" );
         secondPhaseTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
   }
#endif
}

template< typename Value, typename Index, typename Compare, int stackSize >
__global__
void
cudaQuickSortSecondPhase(
   Containers::ArrayView< Value, Devices::Cuda, Index > arr,
   Containers::ArrayView< Value, Devices::Cuda, Index > aux,
   Compare compare,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks,
   int elementsInShared,
   int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = reinterpret_cast< Value* >( externMem );

   QuicksortTask< Index >& myTask = secondPhaseTasks[ blockIdx.x ];
   if( myTask.getSize() <= 0 )
      return;

   auto arrView = arr.getView( myTask.partitionBegin, myTask.partitionEnd );
   auto auxView = aux.getView( myTask.partitionBegin, myTask.partitionEnd );

   if( elementsInShared == 0 ) {
      singleBlockQuickSort< Value, Index, Compare, stackSize, false >(
         arrView, auxView, compare, myTask.iteration, sharedMem, 0, maxBitonicSize );
   }
   else {
      singleBlockQuickSort< Value, Index, Compare, stackSize, true >(
         arrView, auxView, compare, myTask.iteration, sharedMem, elementsInShared, maxBitonicSize );
   }
#endif
}

template< typename Value, typename Index, typename Compare, int stackSize >
__global__
void
cudaQuickSortSecondPhase2(
   Containers::ArrayView< Value, Devices::Cuda, Index > arr,
   Containers::ArrayView< Value, Devices::Cuda, Index > aux,
   Compare compare,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks1,
   Containers::ArrayView< QuicksortTask< Index >, Devices::Cuda > secondPhaseTasks2,
   int elementsInShared,
   int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = reinterpret_cast< Value* >( externMem );

   QuicksortTask< Index > myTask;
   if( blockIdx.x < secondPhaseTasks1.getSize() )
      myTask = secondPhaseTasks1[ blockIdx.x ];
   else
      myTask = secondPhaseTasks2[ blockIdx.x - secondPhaseTasks1.getSize() ];

   if( myTask.getSize() <= 0 )
      return;

   auto arrView = arr.getView( myTask.partitionBegin, myTask.partitionEnd );
   auto auxView = aux.getView( myTask.partitionBegin, myTask.partitionEnd );

   if( elementsInShared <= 0 ) {
      singleBlockQuickSort< Value, Index, Compare, stackSize, false >(
         arrView, auxView, compare, myTask.iteration, sharedMem, 0, maxBitonicSize );
   }
   else {
      singleBlockQuickSort< Value, Index, Compare, stackSize, true >(
         arrView, auxView, compare, myTask.iteration, sharedMem, elementsInShared, maxBitonicSize );
   }
#endif
}

}  // namespace TNL::Algorithms::Sorting::experimental::detail
