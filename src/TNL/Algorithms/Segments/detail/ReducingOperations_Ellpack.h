// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_Ellpack.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization >
struct ReducingOperations< EllpackView< Device, Index, Organization > >
{
   using SegmentsViewType = EllpackView< Device, Index, Organization >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO Fetch&& does not work with nvcc
                   Reduction reduction,  // TODO Reduction&& does not work with nvcc
                   ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const Index segmentsCount = end - begin;
            const Index threadsCount = segmentsCount * Backend::getWarpSize();
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel =
               EllpackCudaReductionKernel< ConstViewType, IndexBegin, IndexEnd, Fetch, Reduction, ResultKeeper, ReturnType >;
            Backend::launchKernelSync( kernel, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
         }
         else {  // CPU
            const IndexType segmentSize = segments.getSegmentSize();
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               ReturnType aux = identity;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  aux =
                     reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
               keeper( segmentIdx, aux );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {  // ColumnMajorOrder
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            ReturnType aux = identity;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j += alignedSize )
               aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
            keeper( segmentIdx, aux );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
