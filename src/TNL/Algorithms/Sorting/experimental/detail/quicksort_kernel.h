// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
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
   Index iteration,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt );

//-----------------------------------------------------------

template< typename Value, typename Index, typename Compare >
__global__
void
cudaInitTask(
   Containers::ArrayView< QuicksortTask, Devices::Cuda > cuda_tasks,
   Containers::ArrayView< int, Devices::Cuda > cuda_blockToTaskMapping,
   Containers::ArrayView< int, Devices::Cuda > cuda_reductionTaskInitMem,
   Containers::ArrayView< Value, Devices::Cuda, Index > src,
   Compare compare )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( blockIdx.x >= cuda_tasks.getSize() )
      return;

   int start = blockIdx.x == 0 ? 0 : cuda_reductionTaskInitMem[ blockIdx.x - 1 ];
   int end = cuda_reductionTaskInitMem[ blockIdx.x ];
   for( int i = start + threadIdx.x; i < end; i += blockDim.x )
      cuda_blockToTaskMapping[ i ] = blockIdx.x;

   if( threadIdx.x == 0 ) {
      QuicksortTask& task = cuda_tasks[ blockIdx.x ];
      int pivotIdx = task.partitionBegin + pickPivotIdx( src.getView( task.partitionBegin, task.partitionEnd ), compare );
      task.init( start, end - start, pivotIdx );
   }
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
   Containers::ArrayView< QuicksortTask, Devices::Cuda > tasks,
   Containers::ArrayView< int, Devices::Cuda > taskMapping )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = (Value*) externMem;
   Value* sharedMem = piv + 1;

   QuicksortTask& myTask = tasks[ taskMapping[ blockIdx.x ] ];
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
   int maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > tasks,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = (Value*) externMem;

   QuicksortTask& myTask = tasks[ blockIdx.x ];

   if( threadIdx.x == 0 )
      *piv = ( myTask.iteration & 1 ) == 0 ? arr[ myTask.pivotIdx ] : aux[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   int leftBegin = myTask.partitionBegin;
   int leftEnd = myTask.partitionBegin + myTask.dstBegin;
   int rightBegin = myTask.partitionBegin + myTask.dstEnd;
   int rightEnd = myTask.partitionEnd;

   for( int i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x )
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
   Index iteration,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks,
   int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   int size = end - begin;
   TNL_ASSERT_GE( size, 0, "negative size in writeNewTask" );

   if( size == 0 )
      return;

   if( size <= maxElementsForSecondPhase ) {
      int idx = atomicAdd( secondPhaseTasksCnt, 1 );
      if( idx < secondPhaseTasks.getSize() )
         secondPhaseTasks[ idx ] = QuicksortTask( begin, end, iteration + 1 );
      else {
         int idx = atomicAdd( newTasksCnt, 1 );
         TNL_ASSERT_LT( idx, newTasks.getSize(), "task memory exhausted in writeNewTask" );
         newTasks[ idx ] = QuicksortTask( begin, end, iteration + 1 );
      }
   }
   else {
      int idx = atomicAdd( newTasksCnt, 1 );
      if( idx < newTasks.getSize() )
         newTasks[ idx ] = QuicksortTask( begin, end, iteration + 1 );
      else {
         int idx = atomicAdd( secondPhaseTasksCnt, 1 );
         TNL_ASSERT_LT( idx, secondPhaseTasks.getSize(), "task memory exhausted in writeNewTask" );
         secondPhaseTasks[ idx ] = QuicksortTask( begin, end, iteration + 1 );
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
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks,
   int elementsInShared,
   int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = (Value*) externMem;

   QuicksortTask& myTask = secondPhaseTasks[ blockIdx.x ];
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
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks1,
   Containers::ArrayView< QuicksortTask, Devices::Cuda > secondPhaseTasks2,
   int elementsInShared,
   int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = (Value*) externMem;

   QuicksortTask myTask;
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
