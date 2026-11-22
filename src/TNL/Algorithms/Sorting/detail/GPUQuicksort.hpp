// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/DiscreteMath.h>
#include <TNL/Functional.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include "GPUQuicksort/QuicksortTask.h"
#include "GPUQuicksort/GPUQuicksortKernelCompleteTasks.h"
#include "GPUQuicksort/GPUQuicksortKernelPartitionTasks.h"
#include "GPUQuicksort/GPUQuicksortKernelPrepareTasks.h"
#include "GPUQuicksort/GPUQuicksortKernelSortTaskLists.h"
#include "GPUQuicksort/GPUQuicksortKernelSortTasks.h"
#include "GPUQuicksort.h"

namespace TNL::Algorithms::Sorting::detail {

// Number of elements sampled to estimate the median of a partitioned range.
constexpr int maxPivotSamples = 64;

template< typename Value, typename Index >
template< typename Array, typename Compare >
void
GPUQuicksort< Value, Index >::sort( Array& arr, const Compare& cmp )
{
   if( arr.getSize() <= 1 ) {
      return;  // nothing to sort
   }
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // for every block there is a bit of shared memory reserved, the actual value can slightly differ

   int sharedReserve = sizeof( int ) * ( 16 + 3 * 32 );
   int maxSharable = Backend::getSharedMemoryPerBlock( Backend::getDevice() ) - sharedReserve;

   int blockDim = 512;  // best case

   /*
    * the goal is to use shared memory as often as possible
    * each thread in a block will process n elements, n==multiplier
    * + 1 reserved for pivot (statically allocating Value type throws weird error, hence it needs to be dynamic)
    *
    * blockDim*multiplier*sizeof(Value) + 1*sizeof(Value) <= maxSharable
    */
   int elementsPerBlock =
      ( maxSharable - sizeof( Value ) ) / sizeof( Value );  // try to use up all of shared memory to store elements
   constexpr int maxBlocks = ( 1 << 20 );
   // the multiplier is capped by quicksortMaxElementsPerThread so that the
   // first-phase partitioning can hold the tile fragment of one thread in
   // registers (see partitionTaskBlock)
   constexpr int maxMultiplier = quicksortMaxElementsPerThread;
   int multiplier = min( elementsPerBlock / blockDim, maxMultiplier );

   if( multiplier <= 0 )  // a block cant store 512 elements, sorting some really big data
   {
      blockDim = 256;  // try to fit 256 elements
      multiplier = min( elementsPerBlock / blockDim, maxMultiplier );

      if( multiplier <= 0 ) {
         // worst case scenario, shared memory cant be utilized at all because of the sheer size of Value
         // sort has to be done with the use of global memory alone

         this->init( arr, maxBlocks, 512, 0, 0 );
         this->performSort( cmp );
         return;
      }
   }

   TNL_ASSERT_LE( (int) ( blockDim * multiplier * sizeof( Value ) ), maxSharable, "" );

   this->init( arr, maxBlocks, blockDim, multiplier * blockDim, maxSharable );
   this->performSort( cmp );
#endif
}

template< typename Value, typename Index >
template< typename Array >
void
GPUQuicksort< Value, Index >::sort( Array& arr )
{
   this->sort( arr, std::less<>{} );
}

template< typename Value, typename Index >
void
GPUQuicksort< Value, Index >::init(
   Containers::ArrayView< Value, Devices::GPU, Index > arr,
   int gridDim,
   int blockDim,
   int desiredElementsPerBlock,
   int maxSharable )
{
   this->maxBlocks = gridDim;
   this->threadsPerBlock = blockDim;
   this->desiredElementsPerBlock = desiredElementsPerBlock;
   this->maxSharable = maxSharable;
   this->arr.bind( arr );
   this->auxMem.setSize( arr.getSize() );
   this->aux.bind( auxMem.getView() );
   this->desiredSecondPhaseElementsPerBlock = desiredElementsPerBlock;
   this->maxTasks = static_cast< int >( min( arr.getSize(), static_cast< Index >( maxTasksLimit ) ) );
   this->taskStorage.setSize( 3 * maxTasks );
   this->tasks.bind( taskStorage.getView( 0, maxTasks ) );
   this->newTasks.bind( taskStorage.getView( maxTasks, 2 * maxTasks ) );
   this->secondPhaseTasks.bind( taskStorage.getView( 2 * maxTasks, 3 * maxTasks ) );
   this->taskCounters.setSize( 2 );
   /*
    * Capacity of the block-to-task map. Every first-phase task holds more than
    * elementsPerBlock elements (smaller children go to the second phase), so
    * there are fewer than size/elementsPerBlock of them and each contributes
    * at most size/elementsPerBlock + 1 blocks. Hence the total block demand of
    * any round stays below 2*ceil(size/elementsPerBlock) + 8, which keeps the
    * premature-second-phase fallback below exactly as selective as with the
    * historical fixed capacity of 2^20 entries, without allocating 4 MB for
    * every sort.
    */
   int blockToTaskCapacity = maxBlocks;
   if( desiredElementsPerBlock > 0 ) {
      const auto maxTiles =
         static_cast< std::size_t >( roundUpDivision( arr.getSize(), static_cast< Index >( desiredElementsPerBlock ) ) );
      blockToTaskCapacity = static_cast< int >( min( static_cast< std::size_t >( maxBlocks ), 2 * maxTiles + 8 ) );
   }
   this->blockToTask.setSize( blockToTaskCapacity );
   this->reductionTaskInitMem.setSize( maxTasks );

   if( arr.getSize() > static_cast< Index >( desiredSecondPhaseElementsPerBlock ) ) {
      tasks.setElement( 0, QuicksortTask< Index >( 0, arr.getSize(), 0 ) );
      host_firstPhaseTasksAmount = 1;
   }
   else {
      secondPhaseTasks.setElement( 0, QuicksortTask< Index >( 0, arr.getSize(), 0 ) );
      host_secondPhaseTasksAmount = 1;
   }

   taskCounters.setValue( 0 );
}

template< typename Value, typename Index >
template< typename Compare >
void
GPUQuicksort< Value, Index >::performSort( const Compare& compare )
{
   firstPhase( compare );

   int totalSecondPhase = host_firstPhaseTasksAmount + host_secondPhaseTasksAmount;
   if( totalSecondPhase > 0 )
      secondPhase( compare );
}

template< typename Value, typename Index >
template< typename Compare >
void
GPUQuicksort< Value, Index >::firstPhase( const Compare& compare )
{
   while( host_firstPhaseTasksAmount > 0 ) {
      if( host_firstPhaseTasksAmount >= maxTasks )
         break;

      if( host_secondPhaseTasksAmount
          >= maxTasks )  // second phase occupies enough tasks to warrant premature second phase sort
      {
         int tmp = host_firstPhaseTasksAmount;
         host_firstPhaseTasksAmount = 0;
         secondPhase( compare );
         taskCounters.setElement( 1, 0 );
         host_secondPhaseTasksAmount = 0;
         host_firstPhaseTasksAmount = tmp;
      }

      // just in case newly created tasks wouldn't fit
      // bite the bullet and sort with single blocks
      if( host_firstPhaseTasksAmount * 2 >= maxTasks + ( maxTasks - host_secondPhaseTasksAmount ) ) {
         if( host_secondPhaseTasksAmount
             >= 0.75 * maxTasks )  // second phase occupies enough tasks to warrant premature second phase sort
         {
            int tmp = host_firstPhaseTasksAmount;
            host_firstPhaseTasksAmount = 0;
            secondPhase( compare );
            taskCounters.setElement( 1, 0 );
            host_secondPhaseTasksAmount = 0;
            host_firstPhaseTasksAmount = tmp;
         }
         else {
            break;
         }
      }

      int elementsPerBlock = getElementsPerBlock();

      /*
       * initializes tasks so that each block knows which task to work on and which part of array to split
       * also sets pivot needed for partitioning, this is why compare is needed
       */
      int blocksCnt = initTasks( elementsPerBlock, compare );

      // not enough or too many blocks needed, switch to second phase
      if( blocksCnt <= 1 || blocksCnt > blockToTask.getSize() )
         break;

      // do the partitioning

      auto& task = iteration % 2 == 0 ? tasks : newTasks;

      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = threadsPerBlock;
      launch_config.gridSize.x = blocksCnt;
      launch_config.dynamicSharedMemorySize = elementsPerBlock * sizeof( Value ) + sizeof( Value );  // elements + 1 for pivot

      /*
       * check if partition procedure can use shared memory for coalesced write after reordering
       *
       * move elements smaller than pivot to the left and bigger to the right
       * note: pivot isnt inserted in the middle yet
       *
       * The partitioning, the pivot fill and the task creation are launched
       * asynchronously and serialized by the stream: the host does not need any
       * of their results until the new task counters are read in processNewTasks,
       * so synchronizing after each of them would only leave the GPU idle.
       */
      if( launch_config.dynamicSharedMemorySize <= maxSharable ) {
         constexpr auto kernel = GPUQuicksortKernelPartitionTasks< Value, Index, Compare, true >;
         Backend::launchKernelAsync( kernel, launch_config, arr, aux, compare, elementsPerBlock, task, blockToTask );
      }
      else {
         launch_config.dynamicSharedMemorySize = sizeof( Value );
         constexpr auto kernel = GPUQuicksortKernelPartitionTasks< Value, Index, Compare, false >;
         Backend::launchKernelAsync( kernel, launch_config, arr, aux, compare, elementsPerBlock, task, blockToTask );
      }

      /*
       * fill in the gap between smaller and bigger with elements == pivot
       * after writing also create new tasks, each task generates at max 2 tasks
       *
       * tasks smaller than desiredSecondPhaseElementsPerBlock go into second phase
       * bigger need more blocks to partition and are written into newTask
       * with iteration %2, rotate between the 2 tasks array to save from copying
       */
      auto& newTask = iteration % 2 == 0 ? newTasks : tasks;
      launch_config.gridSize.x = host_firstPhaseTasksAmount;
      launch_config.dynamicSharedMemorySize = sizeof( Value );
      constexpr auto kernel = GPUQuicksortKernelCompleteTasks< Value, Index >;
      Backend::launchKernelAsync(
         kernel,
         launch_config,
         arr,
         aux,
         static_cast< Index >( desiredSecondPhaseElementsPerBlock ),
         task,
         newTask,
         taskCounters.getData(),
         secondPhaseTasks,
         taskCounters.getData() + 1 );

      // reads the task counters, which also waits for the kernels launched above
      processNewTasks();
      iteration++;
   }
}

template< typename Value, typename Index >
template< typename Compare >
void
GPUQuicksort< Value, Index >::secondPhase( const Compare& compare )
{
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = threadsPerBlock;
   launch_config.gridSize.x = host_firstPhaseTasksAmount + host_secondPhaseTasksAmount;
   constexpr int stackSize = 32;
   auto& leftoverTasks = iteration % 2 == 0 ? tasks : newTasks;

   int elementsInShared = desiredElementsPerBlock;
   launch_config.dynamicSharedMemorySize =
      elementsInShared * sizeof( Value ) + sizeof( Value );  // reserve space for storing elements + 1 pivot
   if( launch_config.dynamicSharedMemorySize > maxSharable ) {
      launch_config.dynamicSharedMemorySize = sizeof( Value );
      elementsInShared = 0;
   }

   if( host_firstPhaseTasksAmount > 0 && host_secondPhaseTasksAmount > 0 ) {
      auto tasks = leftoverTasks.getView( 0, host_firstPhaseTasksAmount );
      auto tasks2 = secondPhaseTasks.getView( 0, host_secondPhaseTasksAmount );

      constexpr auto kernel = GPUQuicksortKernelSortTaskLists< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks, tasks2, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
   else if( host_firstPhaseTasksAmount > 0 ) {
      auto tasks = leftoverTasks.getView( 0, host_firstPhaseTasksAmount );
      constexpr auto kernel = GPUQuicksortKernelSortTasks< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
   else {
      auto tasks2 = secondPhaseTasks.getView( 0, host_secondPhaseTasksAmount );
      constexpr auto kernel = GPUQuicksortKernelSortTasks< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks2, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
}

template< typename Value, typename Index >
int
GPUQuicksort< Value, Index >::getElementsPerBlock() const
{
   return desiredElementsPerBlock;
}

template< typename Value, typename Index >
template< typename Compare >
int
GPUQuicksort< Value, Index >::initTasks( int elementsPerBlock, const Compare& compare )
{
   auto& src = iteration % 2 == 0 ? arr : aux;
   auto& activeTasks = iteration % 2 == 0 ? tasks : newTasks;

   {
      const auto& taskView = activeTasks.getConstView( 0, host_firstPhaseTasksAmount );
      auto blocksNeeded = reductionTaskInitMem.getView( 0, host_firstPhaseTasksAmount );
      parallelFor< Devices::GPU >(
         0,
         host_firstPhaseTasksAmount,
         [ = ] __cuda_callable__( int i ) mutable
         {
            const QuicksortTask< Index >& task = taskView[ i ];
            Index size = task.getSize();
            blocksNeeded[ i ] = TNL::roundUpDivision( size, static_cast< Index >( elementsPerBlock ) );
         } );
   }
   // reductionTaskInitMem[i] == how many blocks task i needs

   // only the active prefix is scanned (and read afterwards): scanning the
   // whole array paid 3.3 us per round even with two active tasks at 1e6
   inplaceInclusiveScan( reductionTaskInitMem, 0, host_firstPhaseTasksAmount );
   // reductionTaskInitMem[i] == how many blocks task [0..i] need

   int blocksNeeded = reductionTaskInitMem.getElement( host_firstPhaseTasksAmount - 1 );

   // need too many blocks, give back control
   if( blocksNeeded > blockToTask.getSize() )
      return blocksNeeded;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = threadsPerBlock;
   launch_config.gridSize.x = host_firstPhaseTasksAmount;
   const int nSamples = TNL::min( threadsPerBlock, maxPivotSamples );
   const std::size_t sampleBufferSize = static_cast< std::size_t >( nSamples ) * sizeof( Value );
   const bool canBufferSamples = sampleBufferSize <= maxSharable;
   launch_config.dynamicSharedMemorySize = canBufferSamples ? static_cast< int >( sampleBufferSize ) : 0;
   // asynchronous on purpose: the partitioning kernel that reads the tasks is
   // launched right after it on the same stream, so there is nothing to wait for
   Backend::launchKernelAsync(
      GPUQuicksortKernelPrepareTasks< Value, Index, Compare >,
      launch_config,
      activeTasks.getView( 0, host_firstPhaseTasksAmount ),           // task to read from
      blockToTask.getView( 0, blocksNeeded ),                         // maps block to a certain task
      reductionTaskInitMem.getView( 0, host_firstPhaseTasksAmount ),  // has how many each task need blocks precalculated
      src,
      compare,  // used to pick pivot
      nSamples,
      canBufferSamples,
      taskCounters.getData() );

   return blocksNeeded;
}

template< typename Value, typename Index >
void
GPUQuicksort< Value, Index >::processNewTasks()
{
   // both counters are read with one synchronous transfer, which also waits
   // for all kernels launched in this round
   int counters[ 2 ];
   Backend::memcpy( counters, taskCounters.getData(), 2 * sizeof( int ), Backend::MemcpyDeviceToHost );
   host_firstPhaseTasksAmount = min( counters[ 0 ], maxTasks );
   host_secondPhaseTasksAmount = min( counters[ 1 ], maxTasks );
}

}  // namespace TNL::Algorithms::Sorting::detail
