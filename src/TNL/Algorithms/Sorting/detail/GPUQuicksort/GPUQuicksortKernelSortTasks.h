// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "QuicksortBlock.h"
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Value, typename Index, typename Compare, int stackSize >
__global__
void
GPUQuicksortKernelSortTasks(
   Containers::ArrayView< Value, Devices::GPU, Index > arr,
   Containers::ArrayView< Value, Devices::GPU, Index > aux,
   Compare compare,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > secondPhaseTasks,
   int elementsInShared,
   int maxBitonicSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* sharedMem = reinterpret_cast< Value* >( externMem );

   QuicksortTask< Index >& myTask = secondPhaseTasks[ blockIdx.x ];
   if( myTask.getSize() <= 0 )
      return;

   auto arrView = arr.getView( myTask.partitionBegin, myTask.partitionEnd );
   auto auxView = aux.getView( myTask.partitionBegin, myTask.partitionEnd );

   if( elementsInShared == 0 ) {
      quicksortBlock< Value, Index, Compare, stackSize, false >(
         arrView, auxView, compare, myTask.iteration, sharedMem, 0, maxBitonicSize );
   }
   else {
      quicksortBlock< Value, Index, Compare, stackSize, true >(
         arrView, auxView, compare, myTask.iteration, sharedMem, elementsInShared, maxBitonicSize );
   }
#endif
}
}  // namespace TNL::Algorithms::Sorting::detail
