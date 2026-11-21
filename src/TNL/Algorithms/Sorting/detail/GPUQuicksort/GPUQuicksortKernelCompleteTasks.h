// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "QuicksortTask.h"
#include "QuicksortTaskEnqueue.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Value, typename Index >
__global__
void
GPUQuicksortKernelCompleteTasks(
   Containers::ArrayView< Value, Devices::GPU, Index > arr,
   Containers::ArrayView< Value, Devices::GPU, Index > aux,
   Index maxElementsForSecondPhase,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > tasks,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > newTasks,
   int* newTasksCnt,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > secondPhaseTasks,
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
      enqueueChildTask(
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
      enqueueChildTask(
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
}  // namespace TNL::Algorithms::Sorting::detail
