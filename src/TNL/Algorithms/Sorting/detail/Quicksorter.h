// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/detail/task.h>

namespace TNL::Algorithms::Sorting {

template< typename Value, typename Device, typename Index >
class Quicksorter;

template< typename Value, typename Index >
class Quicksorter< Value, Devices::Cuda, Index >
{
public:
   using ValueType = Value;
   using DeviceType = Devices::Cuda;
   using IndexType = Index;

   template< typename Array, typename Compare >
   void
   sort( Array& arr, const Compare& cmp );

   template< typename Array >
   void
   sort( Array& arr );

   void
   init( Containers::ArrayView< ValueType, Devices::Cuda, IndexType > arr,
         int gridDim,
         int blockDim,
         int desiredElemPerBlock,
         int maxSharable );

   template< typename CMP >
   void
   performSort( const CMP& Cmp );

   /**
    * returns how many blocks are needed to start sort phase 1 if @param elemPerBlock were to be used
    * */
   [[nodiscard]] int
   getSetsNeeded( int elemPerBlock ) const;

   /**
    * returns the optimal amount of elements per thread needed for phase
    * */
   [[nodiscard]] int
   getElemPerBlock() const;

   /**
    * returns the amount of blocks needed to start phase 1 while also initializing all tasks
    * */
   template< typename CMP >
   int
   initTasks( int elemPerBlock, const CMP& Cmp );

   /**
    * does the 1st phase of Quicksort until out of task memory or each task is small enough
    * for correctness, secondphase method needs to be called to sort each subsequences
    * */
   template< typename CMP >
   void
   firstPhase( const CMP& Cmp );

   /**
    * update necessary variables after 1 phase1 sort
    * */
   void
   processNewTasks();

   /**
    * sorts all leftover tasks
    * */
   template< typename CMP >
   void
   secondPhase( const CMP& Cmp );

protected:
   // kernel config
   int maxBlocks, threadsPerBlock, desiredElemPerBlock;
   std::size_t maxSharable;

   Containers::Array< ValueType, Devices::Cuda, IndexType > auxMem;
   Containers::ArrayView< ValueType, Devices::Cuda, IndexType > arr, aux;

   int desired_2ndPhasElemPerBlock;
   const int g_maxTasks = 1 << 14;
   int maxTasks;

   Containers::Array< TASK, Devices::Cuda > cuda_tasks, cuda_newTasks,
      cuda_2ndPhaseTasks;  // 1 set of 2 rotating tasks and 2nd phase
   Containers::Array< int, Devices::Cuda > cuda_newTasksAmount,
      cuda_2ndPhaseTasksAmount;  // is in reality 1 integer each

   Containers::Array< int, Devices::Cuda > cuda_blockToTaskMapping;
   Containers::Array< int, Devices::Cuda > cuda_reductionTaskInitMem;

   int host_1stPhaseTasksAmount = 0, host_2ndPhaseTasksAmount = 0;
   int iteration = 0;

   template< typename ValueType_, typename IndexType_ >
   friend int
   getSetsNeededFunction( int elemPerBlock, const Quicksorter< ValueType_, Devices::Cuda, IndexType_ >& quicksort );
};

}  // namespace TNL::Algorithms::Sorting

#include <TNL/Algorithms/Sorting/detail/Quicksorter.hpp>
