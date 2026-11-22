// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Allocators/Default.h>
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

   /**
    * \brief One uninitialized device allocation for the two rotating
    * first-phase task sets and the second-phase set; the three views below
    * are bound into it.
    *
    * Unlike \ref Containers::Array::setSize, the storage never constructs or
    * destructs its elements. This is sound because `QuicksortTask` is
    * trivially constructible and trivially destructible (enforced by the
    * static assertions below) and only the task slots claimed by the task
    * counters are ever read - the initial task is written by the host in
    * \ref GPUQuicksort::init, every other slot is written by
    * `enqueueChildTask` when it is claimed, and the helper variables of an
    * active first-phase task are set by `QuicksortTask::init` in
    * `GPUQuicksortKernelPrepareTasks` before the kernels of the same round
    * read them. Value-initializing the whole storage with
    * `Containers::Array` instead launched a placement-new kernel zeroing
    * 3*maxTasks task structs, which was pure overhead: the sorted data never
    * touches most of the slots.
    */
   struct TaskStorage
   {
      using Task = QuicksortTask< IndexType >;
      using Allocator = typename Allocators::Default< Devices::GPU >::template Allocator< Task >;

      static_assert(
         std::is_trivially_default_constructible_v< Task >,
         "task storage skips element construction, so the task type must stay trivially constructible" );
      static_assert(
         std::is_trivially_destructible_v< Task >,
         "task storage skips element destruction, so the task type must stay trivially destructible" );

      TaskStorage() = default;
      TaskStorage( const TaskStorage& ) = delete;
      TaskStorage&
      operator=( const TaskStorage& ) = delete;

      ~TaskStorage()
      {
         if( data != nullptr ) {
            Allocator allocator;
            allocator.deallocate( data, size );
         }
      }

      void
      setSize( int newSize )
      {
         if( newSize == size )
            return;
         if( data != nullptr ) {
            Allocator allocator;
            allocator.deallocate( data, size );
            data = nullptr;
            size = 0;
         }
         if( newSize > 0 ) {
            Allocator allocator;
            data = allocator.allocate( newSize );
            size = newSize;
         }
      }

      [[nodiscard]] Containers::ArrayView< Task, Devices::GPU >
      getView( int begin, int end )
      {
         return { data + begin, end - begin };
      }

      Task* data = nullptr;
      int size = 0;
   } taskStorage;

   Containers::ArrayView< QuicksortTask< IndexType >, Devices::GPU > tasks, newTasks, secondPhaseTasks;
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
