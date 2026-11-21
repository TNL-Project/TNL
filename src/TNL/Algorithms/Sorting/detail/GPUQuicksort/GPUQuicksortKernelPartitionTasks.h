// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "QuicksortTask.h"
#include "QuicksortTaskPartition.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Value, typename Index, typename Compare, bool useShared >
__global__
void
GPUQuicksortKernelPartitionTasks(
   Containers::ArrayView< Value, Devices::GPU, Index > arr,
   Containers::ArrayView< Value, Devices::GPU, Index > aux,
   const Compare& compare,
   int elementsPerBlock,
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > tasks,
   Containers::ArrayView< int, Devices::GPU > taskMapping )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   extern __shared__ int externMem[];
   Value* piv = reinterpret_cast< Value* >( externMem );
   Value* sharedMem = piv + 1;

   QuicksortTask< Index >& myTask = tasks[ taskMapping[ blockIdx.x ] ];
   auto& src = ( myTask.iteration & 1 ) == 0 ? arr : aux;
   auto& dst = ( myTask.iteration & 1 ) == 0 ? aux : arr;

   if( threadIdx.x == 0 )
      *piv = src[ myTask.pivotIdx ];
   __syncthreads();
   Value& pivot = *piv;

   partitionTaskBlock< Value, Index, Compare, useShared >(
      src.getView( myTask.partitionBegin, myTask.partitionEnd ),
      dst.getView( myTask.partitionBegin, myTask.partitionEnd ),
      compare,
      sharedMem,
      pivot,
      elementsPerBlock,
      myTask );
#endif
}
}  // namespace TNL::Algorithms::Sorting::detail
