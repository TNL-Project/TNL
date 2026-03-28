// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/DiscreteMath.h>
#include <TNL/Functional.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include "QuicksortTask.h"
#include "quicksort_kernel.h"
#include "Quicksorter.h"

namespace TNL::Algorithms::Sorting::experimental::detail {

template< typename Value, typename Index >
template< typename Array, typename Compare >
void
Quicksorter< Value, Devices::Cuda, Index >::sort( Array& arr, const Compare& cmp )
{
   if( arr.getSize() <= 1 ) {
      return;  // nothing to sort
   }
#if defined( __CUDACC__ ) || defined( __HIP__ )
   /**
    * for every block there is a bit of shared memory reserved, the actual value can slightly differ
    * */
   int sharedReserve = sizeof( int ) * ( 16 + 3 * 32 );
   int maxSharable = Backend::getSharedMemoryPerBlock( Backend::getDevice() ) - sharedReserve;

   int blockDim = 512;  // best case

   /**
    * the goal is to use shared memory as often as possible
    * each thread in a block will process n elements, n==multiplier
    * + 1 reserved for pivot (statically allocating Value type throws weird error, hence it needs to be dynamic)
    *
    * blockDim*multiplier*sizeof(Value) + 1*sizeof(Value) <= maxSharable
    * */
   int elementsPerBlock =
      ( maxSharable - sizeof( Value ) ) / sizeof( Value );  // try to use up all of shared memory to store elements
   constexpr int maxBlocks = ( 1 << 20 );
   constexpr int maxMultiplier = 8;
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
Quicksorter< Value, Devices::Cuda, Index >::sort( Array& arr )
{
   this->sort( arr, std::less<>{} );
}

template< typename Value, typename Index >
void
Quicksorter< Value, Devices::Cuda, Index >::init(
   Containers::ArrayView< Value, Devices::Cuda, Index > arr,
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
   this->maxTasks = min( arr.getSize(), maxTasksLimit );
   this->cuda_tasks.setSize( maxTasks );
   this->cuda_newTasks.setSize( maxTasks );
   this->cuda_secondPhaseTasks.setSize( maxTasks );
   this->cuda_newTasksAmount.setSize( 1 );
   this->cuda_secondPhaseTasksAmount.setSize( 1 );
   this->cuda_blockToTaskMapping.setSize( maxBlocks );
   this->cuda_reductionTaskInitMem.setSize( maxTasks );

   if( arr.getSize() > static_cast< Index >( desiredSecondPhaseElementsPerBlock ) ) {
      cuda_tasks.setElement( 0, QuicksortTask( 0, arr.getSize(), 0 ) );
      host_firstPhaseTasksAmount = 1;
   }
   else {
      cuda_secondPhaseTasks.setElement( 0, QuicksortTask( 0, arr.getSize(), 0 ) );
      host_secondPhaseTasksAmount = 1;
   }

   cuda_secondPhaseTasksAmount = 0;
}

template< typename Value, typename Index >
template< typename Compare >
void
Quicksorter< Value, Devices::Cuda, Index >::performSort( const Compare& compare )
{
   firstPhase( compare );

   int totalSecondPhase = host_firstPhaseTasksAmount + host_secondPhaseTasksAmount;
   if( totalSecondPhase > 0 )
      secondPhase( compare );
}

template< typename Value, typename Index >
template< typename Compare >
void
Quicksorter< Value, Devices::Cuda, Index >::firstPhase( const Compare& compare )
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
         cuda_secondPhaseTasksAmount = host_secondPhaseTasksAmount = 0;
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
            cuda_secondPhaseTasksAmount = host_secondPhaseTasksAmount = 0;
            host_firstPhaseTasksAmount = tmp;
         }
         else {
            break;
         }
      }

      int elementsPerBlock = getElementsPerBlock();

      /**
       * initializes tasks so that each block knows which task to work on and which part of array to split
       * also sets pivot needed for partitioning, this is why compare is needed
       * */
      int blocksCnt = initTasks( elementsPerBlock, compare );

      // not enough or too many blocks needed, switch to second phase
      if( blocksCnt <= 1 || blocksCnt > cuda_blockToTaskMapping.getSize() )
         break;

      // do the partitioning

      auto& task = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = threadsPerBlock;
      launch_config.gridSize.x = blocksCnt;
      launch_config.dynamicSharedMemorySize = elementsPerBlock * sizeof( Value ) + sizeof( Value );  // elements + 1 for pivot

      /**
       * check if partition procedure can use shared memory for coalesced write after reordering
       *
       * move elements smaller than pivot to the left and bigger to the right
       * note: pivot isnt inserted in the middle yet
       * */
      if( launch_config.dynamicSharedMemorySize <= maxSharable ) {
         constexpr auto kernel = cudaQuickSortFirstPhase< Value, Index, Compare, true >;
         Backend::launchKernelSync( kernel, launch_config, arr, aux, compare, elementsPerBlock, task, cuda_blockToTaskMapping );
      }
      else {
         launch_config.dynamicSharedMemorySize = sizeof( Value );
         constexpr auto kernel = cudaQuickSortFirstPhase< Value, Index, Compare, false >;
         Backend::launchKernelSync( kernel, launch_config, arr, aux, compare, elementsPerBlock, task, cuda_blockToTaskMapping );
      }

      /**
       * fill in the gap between smaller and bigger with elements == pivot
       * after writing also create new tasks, each task generates at max 2 tasks
       *
       * tasks smaller than desiredSecondPhaseElementsPerBlock go into second phase
       * bigger need more blocks to partition and are written into newTask
       * with iteration %2, rotate between the 2 tasks array to save from copying
       * */
      auto& newTask = iteration % 2 == 0 ? cuda_newTasks : cuda_tasks;
      launch_config.gridSize.x = host_firstPhaseTasksAmount;
      launch_config.dynamicSharedMemorySize = sizeof( Value );
      constexpr auto kernel = cudaWritePivot< Value, Index >;
      Backend::launchKernelSync(
         kernel,
         launch_config,
         arr,
         aux,
         desiredSecondPhaseElementsPerBlock,
         task,
         newTask,
         cuda_newTasksAmount.getData(),
         cuda_secondPhaseTasks,
         cuda_secondPhaseTasksAmount.getData() );

      processNewTasks();
      iteration++;
   }
}

template< typename Value, typename Index >
template< typename Compare >
void
Quicksorter< Value, Devices::Cuda, Index >::secondPhase( const Compare& compare )
{
   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = threadsPerBlock;
   launch_config.gridSize.x = host_firstPhaseTasksAmount + host_secondPhaseTasksAmount;
   constexpr int stackSize = 32;
   auto& leftoverTasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

   int elementsInShared = desiredElementsPerBlock;
   launch_config.dynamicSharedMemorySize =
      elementsInShared * sizeof( Value ) + sizeof( Value );  // reserve space for storing elements + 1 pivot
   if( launch_config.dynamicSharedMemorySize > maxSharable ) {
      launch_config.dynamicSharedMemorySize = sizeof( Value );
      elementsInShared = 0;
   }

   if( host_firstPhaseTasksAmount > 0 && host_secondPhaseTasksAmount > 0 ) {
      auto tasks = leftoverTasks.getView( 0, host_firstPhaseTasksAmount );
      auto tasks2 = cuda_secondPhaseTasks.getView( 0, host_secondPhaseTasksAmount );

      constexpr auto kernel = cudaQuickSortSecondPhase2< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks, tasks2, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
   else if( host_firstPhaseTasksAmount > 0 ) {
      auto tasks = leftoverTasks.getView( 0, host_firstPhaseTasksAmount );
      constexpr auto kernel = cudaQuickSortSecondPhase< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
   else {
      auto tasks2 = cuda_secondPhaseTasks.getView( 0, host_secondPhaseTasksAmount );
      constexpr auto kernel = cudaQuickSortSecondPhase< Value, Index, Compare, stackSize >;
      Backend::launchKernelSync(
         kernel, launch_config, arr, aux, compare, tasks2, elementsInShared, desiredSecondPhaseElementsPerBlock );
   }
}

template< typename Value, typename Index >
int
Quicksorter< Value, Devices::Cuda, Index >::getElementsPerBlock() const
{
   return desiredElementsPerBlock;
}

template< typename Value, typename Index >
template< typename Compare >
int
Quicksorter< Value, Devices::Cuda, Index >::initTasks( int elementsPerBlock, const Compare& compare )
{
   auto& src = iteration % 2 == 0 ? arr : aux;
   auto& tasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

   {
      const auto& cuda_tasks = tasks.getConstView( 0, host_firstPhaseTasksAmount );
      auto blocksNeeded = cuda_reductionTaskInitMem.getView( 0, host_firstPhaseTasksAmount );
      parallelFor< Devices::Cuda >(
         0,
         host_firstPhaseTasksAmount,
         [ = ] __cuda_callable__( int i ) mutable
         {
            const QuicksortTask& task = cuda_tasks[ i ];
            int size = task.getSize();
            blocksNeeded[ i ] = TNL::roundUpDivision( size, elementsPerBlock );
         } );
   }
   // cuda_reductionTaskInitMem[i] == how many blocks task i needs

   inplaceInclusiveScan( cuda_reductionTaskInitMem );
   // cuda_reductionTaskInitMem[i] == how many blocks task [0..i] need

   int blocksNeeded = cuda_reductionTaskInitMem.getElement( host_firstPhaseTasksAmount - 1 );

   // need too many blocks, give back control
   if( blocksNeeded > cuda_blockToTaskMapping.getSize() )
      return blocksNeeded;

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = threadsPerBlock;
   launch_config.gridSize.x = host_firstPhaseTasksAmount;
   Backend::launchKernelSync(
      cudaInitTask< Value, Index, Compare >,
      launch_config,
      tasks.getView( 0, host_firstPhaseTasksAmount ),                      // task to read from
      cuda_blockToTaskMapping.getView( 0, blocksNeeded ),                  // maps block to a certain task
      cuda_reductionTaskInitMem.getView( 0, host_firstPhaseTasksAmount ),  // has how many each task need blocks precalculated
      src,
      compare );  // used to pick pivot

   cuda_newTasksAmount.setElement( 0, 0 );  // resets new element counter
   return blocksNeeded;
}

template< typename Value, typename Index >
void
Quicksorter< Value, Devices::Cuda, Index >::processNewTasks()
{
   host_firstPhaseTasksAmount = min( cuda_newTasksAmount.getElement( 0 ), maxTasks );
   host_secondPhaseTasksAmount = min( cuda_secondPhaseTasksAmount.getElement( 0 ), maxTasks );
}

}  // namespace TNL::Algorithms::Sorting::experimental::detail
