// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include "task.h"
#include "cudaPartition.h"
#include "quicksort_1Block.h"

namespace TNL::Algorithms::Sorting::experimental::detail {

template< typename Index >
__device__
void
writeNewTask( Index begin,
              Index end,
              Index iteration,
              Index maxElemFor2ndPhase,
              Containers::ArrayView< TASK, Devices::Cuda > newTasks,
              int* newTasksCnt,
              Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks,
              int* secondPhaseTasksCnt );

//-----------------------------------------------------------

template< typename Value, typename CMP >
__global__
void
cudaInitTask( Containers::ArrayView< TASK, Devices::Cuda > cuda_tasks,
              Containers::ArrayView< int, Devices::Cuda > cuda_blockToTaskMapping,
              Containers::ArrayView< int, Devices::Cuda > cuda_reductionTaskInitMem,
              Containers::ArrayView< Value, Devices::Cuda > src,
              CMP Cmp )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( blockIdx.x >= cuda_tasks.getSize() )
      return;

   int start = blockIdx.x == 0 ? 0 : cuda_reductionTaskInitMem[ blockIdx.x - 1 ];
   int end = cuda_reductionTaskInitMem[ blockIdx.x ];
   for( int i = start + threadIdx.x; i < end; i += blockDim.x )
      cuda_blockToTaskMapping[ i ] = blockIdx.x;

   if( threadIdx.x == 0 ) {
      TASK& task = cuda_tasks[ blockIdx.x ];
      int pivotIdx = task.partitionBegin + pickPivotIdx( src.getView( task.partitionBegin, task.partitionEnd ), Cmp );
      task.initTask( start, end - start, pivotIdx );
   }
#endif
}

//----------------------------------------------------

template< typename Value, typename CMP, bool useShared >
__global__
void
cudaQuickSort1stPhase( Containers::ArrayView< Value, Devices::Cuda > arr,
                       Containers::ArrayView< Value, Devices::Cuda > aux,
                       const CMP& Cmp,
                       int elemPerBlock,
                       Containers::ArrayView< TASK, Devices::Cuda > tasks,
                       Containers::ArrayView< int, Devices::Cuda > taskMapping )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = (Value*) externMem;
   Value* sharedMem = piv + 1;

   TASK& myTask = tasks[ taskMapping[ blockIdx.x ] ];
   auto& src = ( myTask.iteration & 1 ) == 0 ? arr : aux;
   auto& dst = ( myTask.iteration & 1 ) == 0 ? aux : arr;

   if( threadIdx.x == 0 )
      *piv = src[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   cudaPartition< Value, CMP, useShared >( src.getView( myTask.partitionBegin, myTask.partitionEnd ),
                                           dst.getView( myTask.partitionBegin, myTask.partitionEnd ),
                                           Cmp,
                                           sharedMem,
                                           pivot,
                                           elemPerBlock,
                                           myTask );
#endif
}

//----------------------------------------------------

template< typename Value >
__global__
void
cudaWritePivot( Containers::ArrayView< Value, Devices::Cuda > arr,
                Containers::ArrayView< Value, Devices::Cuda > aux,
                int maxElemFor2ndPhase,
                Containers::ArrayView< TASK, Devices::Cuda > tasks,
                Containers::ArrayView< TASK, Devices::Cuda > newTasks,
                int* newTasksCnt,
                Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks,
                int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = (Value*) externMem;

   TASK& myTask = tasks[ blockIdx.x ];

   if( threadIdx.x == 0 )
      *piv = ( myTask.iteration & 1 ) == 0 ? arr[ myTask.pivotIdx ] : aux[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   int leftBegin = myTask.partitionBegin;
   int leftEnd = myTask.partitionBegin + myTask.dstBegin;
   int rightBegin = myTask.partitionBegin + myTask.dstEnd;
   int rightEnd = myTask.partitionEnd;

   for( int i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x ) {
      /*
      #ifdef DEBUG
      aux[i] = -1;
      #endif
      */
      arr[ i ] = pivot;
   }

   if( threadIdx.x != 0 )
      return;

   if( leftEnd - leftBegin > 0 ) {
      writeNewTask( leftBegin,
                    leftEnd,
                    myTask.iteration,
                    maxElemFor2ndPhase,
                    newTasks,
                    newTasksCnt,
                    secondPhaseTasks,
                    secondPhaseTasksCnt );
   }

   if( rightEnd - rightBegin > 0 ) {
      writeNewTask( rightBegin,
                    rightEnd,
                    myTask.iteration,
                    maxElemFor2ndPhase,
                    newTasks,
                    newTasksCnt,
                    secondPhaseTasks,
                    secondPhaseTasksCnt );
   }
#endif
}

//-----------------------------------------------------------

template< typename Index >
__device__
void
writeNewTask( Index begin,
              Index end,
              Index iteration,
              Index maxElemFor2ndPhase,
              Containers::ArrayView< TASK, Devices::Cuda > newTasks,
              int* newTasksCnt,
              Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks,
              int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   int size = end - begin;
   if( size < 0 ) {
      printf( "negative size, something went really wrong\n" );
      return;
   }

   if( size == 0 )
      return;

   if( size <= maxElemFor2ndPhase ) {
      int idx = atomicAdd( secondPhaseTasksCnt, 1 );
      if( idx < secondPhaseTasks.getSize() )
         secondPhaseTasks[ idx ] = TASK( begin, end, iteration + 1 );
      else {
         // printf("ran out of memory, trying backup\n");
         int idx = atomicAdd( newTasksCnt, 1 );
         if( idx < newTasks.getSize() )
            newTasks[ idx ] = TASK( begin, end, iteration + 1 );
         else
            printf( "ran out of memory for second phase task, there isnt even space in newTask list\nPart of array may stay "
                    "unsorted!!!\n" );
      }
   }
   else {
      int idx = atomicAdd( newTasksCnt, 1 );
      if( idx < newTasks.getSize() )
         newTasks[ idx ] = TASK( begin, end, iteration + 1 );
      else {
         // printf("ran out of memory, trying backup\n");
         int idx = atomicAdd( secondPhaseTasksCnt, 1 );
         if( idx < secondPhaseTasks.getSize() )
            secondPhaseTasks[ idx ] = TASK( begin, end, iteration + 1 );
         else
            printf( "ran out of memory for newtask, there isnt even space in second phase task list\nPart of array may stay "
                    "unsorted!!!\n" );
      }
   }
#endif
}

//-----------------------------------------------------------

template< typename Value, typename CMP, int stackSize >
__global__
void
cudaQuickSort2ndPhase( Containers::ArrayView< Value, Devices::Cuda > arr,
                       Containers::ArrayView< Value, Devices::Cuda > aux,
                       CMP Cmp,
                       Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks,
                       int elemInShared,
                       int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = (Value*) externMem;

   TASK& myTask = secondPhaseTasks[ blockIdx.x ];
   if( myTask.partitionEnd - myTask.partitionBegin <= 0 ) {
      // printf("empty task???\n");
      return;
   }

   auto arrView = arr.getView( myTask.partitionBegin, myTask.partitionEnd );
   auto auxView = aux.getView( myTask.partitionBegin, myTask.partitionEnd );

   if( elemInShared == 0 ) {
      singleBlockQuickSort< Value, CMP, stackSize, false >(
         arrView, auxView, Cmp, myTask.iteration, sharedMem, 0, maxBitonicSize );
   }
   else {
      singleBlockQuickSort< Value, CMP, stackSize, true >(
         arrView, auxView, Cmp, myTask.iteration, sharedMem, elemInShared, maxBitonicSize );
   }
#endif
}

template< typename Value, typename CMP, int stackSize >
__global__
void
cudaQuickSort2ndPhase2( Containers::ArrayView< Value, Devices::Cuda > arr,
                        Containers::ArrayView< Value, Devices::Cuda > aux,
                        CMP Cmp,
                        Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks1,
                        Containers::ArrayView< TASK, Devices::Cuda > secondPhaseTasks2,
                        int elemInShared,
                        int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = (Value*) externMem;

   TASK myTask;
   if( blockIdx.x < secondPhaseTasks1.getSize() )
      myTask = secondPhaseTasks1[ blockIdx.x ];
   else
      myTask = secondPhaseTasks2[ blockIdx.x - secondPhaseTasks1.getSize() ];

   if( myTask.partitionEnd - myTask.partitionBegin <= 0 ) {
      // printf("empty task???\n");
      return;
   }

   auto arrView = arr.getView( myTask.partitionBegin, myTask.partitionEnd );
   auto auxView = aux.getView( myTask.partitionBegin, myTask.partitionEnd );

   if( elemInShared <= 0 ) {
      singleBlockQuickSort< Value, CMP, stackSize, false >(
         arrView, auxView, Cmp, myTask.iteration, sharedMem, 0, maxBitonicSize );
   }
   else {
      singleBlockQuickSort< Value, CMP, stackSize, true >(
         arrView, auxView, Cmp, myTask.iteration, sharedMem, elemInShared, maxBitonicSize );
   }
#endif
}

}  // namespace TNL::Algorithms::Sorting::experimental::detail
