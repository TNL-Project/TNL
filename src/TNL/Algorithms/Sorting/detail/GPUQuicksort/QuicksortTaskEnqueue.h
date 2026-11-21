// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Atomic.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Index >
__device__
void
enqueueChildTask(
   Index begin,
   Index end,
   int iteration,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > secondPhaseTasks,
   int* secondPhaseTasksCnt )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Index size = end - begin;
   TNL_ASSERT_GE( size, 0, "negative size in enqueueChildTask" );

   if( size == 0 )
      return;

   if( size <= maxElementsForSecondPhase ) {
      int idx = atomicAdd( secondPhaseTasksCnt, 1 );
      if( idx < secondPhaseTasks.getSize() ) {
         secondPhaseTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
      else {
         int idx = atomicAdd( newTasksCnt, 1 );
         TNL_ASSERT_LT( idx, newTasks.getSize(), "task memory exhausted in enqueueChildTask" );
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
         TNL_ASSERT_LT( idx, secondPhaseTasks.getSize(), "task memory exhausted in enqueueChildTask" );
         secondPhaseTasks[ idx ] = QuicksortTask< Index >( begin, end, iteration + 1 );
      }
   }
#endif
}
}  // namespace TNL::Algorithms::Sorting::detail
