// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/detail/PivotPartitionBlock.h>
#include <TNL/Atomic.h>
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

template< typename Value, typename Index, typename Compare, bool useShared >
__device__
void
partitionTaskBlock(
   Containers::ArrayView< Value, Devices::GPU, Index > src,
   Containers::ArrayView< Value, Devices::GPU, Index > dst,
   const Compare& compare,
   Value* sharedMem,
   const Value& pivot,
   int elementsPerBlock,
   QuicksortTask< Index >& task )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Offset = typename QuicksortTask< Index >::Offset;

   static __shared__ Offset smallerStart;
   static __shared__ Offset biggerStart;

   Index myBegin = static_cast< Index >( elementsPerBlock ) * ( blockIdx.x - task.firstBlock );
   Index myEnd = TNL::min( myBegin + elementsPerBlock, src.getSize() );

   auto srcView = src.getView( myBegin, myEnd );

   const auto scan =
      Algorithms::detail::scanPivotPartitionBlock< Algorithms::detail::PivotPartitionEqualPolicy::FillLater, int >(
         srcView, pivot, compare );

   if( threadIdx.x == blockDim.x - 1 )  // last thread in block has sum of all values
   {
      smallerStart = atomicAdd( &( task.dstBegin ), static_cast< Offset >( scan.inclusive.less ) );
      biggerStart = atomicAdd( &( task.dstEnd ), -static_cast< Offset >( scan.inclusive.greater ) )
                  - static_cast< Offset >( scan.inclusive.greater );
   }
   __syncthreads();

   Algorithms::detail::scatterPivotPartitionBlock< Algorithms::detail::PivotPartitionEqualPolicy::FillLater, useShared >(
      srcView,
      dst,
      pivot,
      compare,
      scan,
      static_cast< Index >( smallerStart ),
      0,
      static_cast< Index >( biggerStart ),
      sharedMem );
#endif
}

}  // namespace TNL::Algorithms::Sorting::detail
