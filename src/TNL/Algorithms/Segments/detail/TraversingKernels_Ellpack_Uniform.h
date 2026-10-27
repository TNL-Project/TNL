// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

template< typename SegmentsView, typename ArrayView, typename Index, typename Function, ElementsOrganization Organization >
__global__
void
forElements_Ellpack_Uniform_WithIndexes(
   const Index gridIdx,
   const std::size_t totalThreadsCount,
   const Index threadsPerSegment,
   SegmentsView segments,
   ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index segmentSize;
   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   __syncthreads();
   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx_idx >= segmentIndexes.getSize() )
         return;

      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index globalIdx = segmentIdx * segmentSize + laneIdx;
         for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            globalIdx += threadsPerSegment;
         }
      }
      else {  // callableArgumentCount< Function >() == 2
         const Index endIdx = ( segmentIdx + 1 ) * segmentSize;
         for( Index globalIdx = segmentIdx * segmentSize + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
   else {  // ColumnMajorOrder
      const Index segmentsCount = segmentIndexes.getSize();
      const Index elementsCount = segmentsCount * segmentSize;
      for( std::size_t globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx );
           globalThreadIdx < static_cast< std::size_t >( elementsCount );
           globalThreadIdx += totalThreadsCount )
      {
         const Index segmentIdx = segmentIndexes[ globalThreadIdx / segmentSize ];
         const Index localIdx = globalThreadIdx % segmentSize;
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }
   }
#endif
}

template<
   typename SegmentsView,
   typename Index,
   typename Condition,
   typename Function,
   ElementsOrganization Organization,
   int BlockSize = 256 >
__global__
void
forElements_Ellpack_Uniform_If(
   const Index gridIdx,
   const std::size_t totalThreadsCount,
   const Index threadsPerSegment,
   SegmentsView segments,
   const Index begin,
   const Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index segmentSize;
   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   __syncthreads();
   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx >= end || ! condition( segmentIdx ) )
         return;

      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index globalIdx = segmentIdx * segmentSize + laneIdx;
         for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            globalIdx += threadsPerSegment;
         }
      }
      else {  // callableArgumentCount< Function >() == 2
         const Index endIdx = ( segmentIdx + 1 ) * segmentSize;
         for( Index globalIdx = segmentIdx * segmentSize + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
   else {  // ColumnMajorOrder
      const Index segmentsCount = ( end - begin );
      const Index elementsCount = segmentsCount * segmentSize;
      for( std::size_t globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx );
           globalThreadIdx < static_cast< std::size_t >( elementsCount );
           globalThreadIdx += totalThreadsCount )
      {
         const Index segmentIdx = globalThreadIdx / segmentSize + begin;
         if( condition( segmentIdx ) ) {
            const Index localIdx = globalThreadIdx % segmentSize;
            const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( callableArgumentCount< Function >() == 3 )
               function( segmentIdx, localIdx, globalIdx );
            else
               function( segmentIdx, globalIdx );
         }
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
