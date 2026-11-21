// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "GPUQuicksort/QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Value, typename Index >
class GPUQuicksort
{
public:
   using ValueType = Value;
   using DeviceType = Devices::GPU;
   using IndexType = Index;

   template< typename Array, typename Compare >
   void
   sort( Array& arr, const Compare& cmp );

   template< typename Array >
   void
   sort( Array& arr );

   void
   init(
      Containers::ArrayView< ValueType, Devices::GPU, IndexType > arr,
      int gridDim,
      int blockDim,
      int desiredElementsPerBlock,
      int maxSharable );

   template< typename Compare >
   void
   performSort( const Compare& compare );

   /**
    * \brief Returns the optimal amount of elements per thread needed for phase 1.
    */
   [[nodiscard]] int
   getElementsPerBlock() const;

   /**
    * \brief Returns the amount of blocks needed to start phase 1 while also initializing all tasks.
    */
   template< typename Compare >
   int
   initTasks( int elementsPerBlock, const Compare& compare );

   /**
    * \brief Does the first phase of Quicksort until out of task memory or each task is small enough.
    *
    * For correctness, the secondPhase method needs to be called to sort each subsequence.
    */
   template< typename Compare >
   void
   firstPhase( const Compare& compare );

   /**
    * \brief Updates necessary variables after one phase 1 sort.
    */
   void
   processNewTasks();

   /**
    * \brief Sorts all leftover tasks.
    */
   template< typename Compare >
   void
   secondPhase( const Compare& compare );

protected:
   // kernel config
   int maxBlocks, threadsPerBlock, desiredElementsPerBlock;
   std::size_t maxSharable;

   Containers::Array< ValueType, Devices::GPU, IndexType > auxMem;
   Containers::ArrayView< ValueType, Devices::GPU, IndexType > arr, aux;

   int desiredSecondPhaseElementsPerBlock;
   static constexpr int maxTasksLimit = 1 << 14;
   int maxTasks;

   Containers::Array< QuicksortTask< IndexType >, Devices::GPU > tasks, newTasks,
      secondPhaseTasks;  // one set of two rotating tasks and second phase
   // task counters, [0] = first-phase tasks of the next round, [1] = second
   // phase tasks; kept in one array so that the host reads both with a
   // single device-to-host transfer
   Containers::Array< int, Devices::GPU > taskCounters;

   Containers::Array< int, Devices::GPU > blockToTask;
   Containers::Array< int, Devices::GPU > reductionTaskInitMem;

   int host_firstPhaseTasksAmount = 0, host_secondPhaseTasksAmount = 0;
   int iteration = 0;
};

}  // namespace TNL::Algorithms::Sorting::detail

#include "GPUQuicksort.hpp"
