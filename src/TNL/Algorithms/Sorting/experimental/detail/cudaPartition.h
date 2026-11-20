// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/detail/PivotPartitionBlock.h>
#include <TNL/Atomic.h>
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::experimental::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Value, typename Index, typename Device, typename Compare >
__device__
std::pair< Value, Index >
pickPivot( TNL::Containers::ArrayView< Value, Device, Index > src, const Compare& compare )
{
   if( src.getSize() <= 1 )
      return { src[ 0 ], 0 };

   const Value& a = src[ 0 ];
   const Value& b = src[ src.getSize() / 2 ];
   const Value& c = src[ src.getSize() - 1 ];

   if( compare( a, b ) )  // ..a..b..
   {
      if( compare( b, c ) )  // ..a..b..c
         return { b, src.getSize() / 2 };
      else if( compare( c, a ) )  //..c..a..b..
         return { a, 0 };
      else  //..a..c..b..
         return { c, src.getSize() - 1 };
   }
   else  //..b..a..
   {
      if( compare( a, c ) )  //..b..a..c
         return { a, 0 };
      else if( compare( c, b ) )  //..c..b..a..
         return { b, src.getSize() / 2 };
      else  //..b..c..a..
         return { c, src.getSize() - 1 };
   }
}

template< typename Value, typename Index, typename Compare, bool useShared >
__device__
void
cudaPartition(
   Containers::ArrayView< Value, Devices::Cuda, Index > src,
   Containers::ArrayView< Value, Devices::Cuda, Index > dst,
   const Compare& compare,
   Value* sharedMem,
   const Value& pivot,
   int elementsPerBlock,
   QuicksortTask< Index >& task )
{
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
}

#endif

}  // namespace TNL::Algorithms::Sorting::experimental::detail
