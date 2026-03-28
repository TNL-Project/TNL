// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::experimental::detail {

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
   init(
      Containers::ArrayView< ValueType, Devices::Cuda, IndexType > arr,
      int gridDim,
      int blockDim,
      int desiredElementsPerBlock,
      int maxSharable );

   template< typename Compare >
   void
   performSort( const Compare& compare );

   /**
    * returns how many blocks are needed to start sort first phase if @param elementsPerBlock were to be used
    * */
   [[nodiscard]] int
   getSetsNeeded( int elementsPerBlock ) const;

   /**
    * returns the optimal amount of elements per thread needed for phase
    * */
   [[nodiscard]] int
   getElementsPerBlock() const;

   /**
    * returns the amount of blocks needed to start phase 1 while also initializing all tasks
    * */
   template< typename Compare >
   int
   initTasks( int elementsPerBlock, const Compare& compare );

   /**
    * does the first phase of Quicksort until out of task memory or each task is small enough
    * for correctness, secondPhase method needs to be called to sort each subsequences
    * */
   template< typename Compare >
   void
   firstPhase( const Compare& compare );

   /**
    * update necessary variables after 1 phase1 sort
    * */
   void
   processNewTasks();

   /**
    * sorts all leftover tasks
    * */
   template< typename Compare >
   void
   secondPhase( const Compare& compare );

protected:
   // kernel config
   int maxBlocks, threadsPerBlock, desiredElementsPerBlock;
   std::size_t maxSharable;

   Containers::Array< ValueType, Devices::Cuda, IndexType > auxMem;
   Containers::ArrayView< ValueType, Devices::Cuda, IndexType > arr, aux;

   int desiredSecondPhaseElementsPerBlock;
   static constexpr int maxTasksLimit = 1 << 14;
   int maxTasks;

   Containers::Array< QuicksortTask, Devices::Cuda > cuda_tasks, cuda_newTasks,
      cuda_secondPhaseTasks;  // one set of two rotating tasks and second phase
   Containers::Array< int, Devices::Cuda > cuda_newTasksAmount,
      cuda_secondPhaseTasksAmount;  // is in reality 1 integer each

   Containers::Array< int, Devices::Cuda > cuda_blockToTaskMapping;
   Containers::Array< int, Devices::Cuda > cuda_reductionTaskInitMem;

   int host_firstPhaseTasksAmount = 0, host_secondPhaseTasksAmount = 0;
   int iteration = 0;

   template< typename ValueType_, typename IndexType_ >
   friend int
   getSetsNeededFunction( int elementsPerBlock, const Quicksorter< ValueType_, Devices::Cuda, IndexType_ >& quicksort );
};

}  // namespace TNL::Algorithms::Sorting::experimental::detail

#include "Quicksorter.hpp"
