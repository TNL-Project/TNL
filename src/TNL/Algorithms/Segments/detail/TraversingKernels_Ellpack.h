// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

//#define USE_CUB
#ifdef __CUDACC__
   #include <cub/cub.cuh>
#endif

namespace TNL::Algorithms::Segments::detail {

template< typename SegmentsView, typename Index, typename Function, ElementsOrganization Organization, int BlockSize = 256 >
__global__
void
forElementsBlockMergeKernel_Ellpack( Index gridIdx, SegmentsView segments, Index begin, Index end, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   if( segments.getSegmentSize() == 0 )
      return;
   const Index globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx );
   if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
      const Index localIdx = globalThreadIdx % segments.getSegmentSize();
      const Index segmentIdx = begin + globalThreadIdx / segments.getSegmentSize();
      if( segmentIdx >= end )
         return;
      TNL_ASSERT_GE( segmentIdx, 0, "" );
      TNL_ASSERT_GE( localIdx, 0, "" );
      TNL_ASSERT_LT( localIdx, segments.getSegmentSize(), "" );
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }
   else {  // ColumnMajorOrder
      const Index segmentsCount = ( end - begin );
      const Index localIdx = globalThreadIdx / segmentsCount;
      const Index segmentIdx = begin + globalThreadIdx % segmentsCount;
      if( localIdx >= segments.getSegmentSize() )
         return;

      TNL_ASSERT_LT( segmentIdx, end, "" );
      TNL_ASSERT_LT( localIdx, segments.getSegmentSize(), "" );
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }
#endif
}

template< typename SegmentsView, typename Index, typename Function, ElementsOrganization Organization >
__global__
void
forElementsKernel_Ellpack( const Index gridIdx,
                           const Index totalThreadsCount,
                           const Index threadsPerSegment,
                           SegmentsView segments,
                           const Index begin,
                           const Index end,
                           Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index segmentSize;
   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   __syncthreads();
   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx >= end )
         return;

      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      if constexpr( argumentCount< Function >() == 3 ) {
         Index globalIdx = segmentIdx * segmentSize + laneIdx;
         for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            globalIdx += threadsPerSegment;
         }
      }
      else {  // argumentCount< Function >() == 2
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
      for( Index globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx ); globalThreadIdx < elementsCount;
           globalThreadIdx += totalThreadsCount )
      {
         const Index segmentIdx = globalThreadIdx / segmentSize + begin;
         const Index localIdx = globalThreadIdx % segmentSize;
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }
   }
#endif
}

template< typename SegmentsView, typename ArrayView, typename Index, typename Function, ElementsOrganization Organization >
__global__
void
forElementsWithSegmentIndexesKernel_Ellpack( const Index gridIdx,
                                             const Index totalThreadsCount,
                                             const Index threadsPerSegment,
                                             SegmentsView segments,
                                             ArrayView segmentIndexes,
                                             const Index begin,
                                             const Index end,
                                             Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index segmentSize;
   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   __syncthreads();
   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx_idx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx_idx >= end )
         return;

      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      if constexpr( argumentCount< Function >() == 3 ) {
         Index globalIdx = segmentIdx * segmentSize + laneIdx;
         for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            globalIdx += threadsPerSegment;
         }
      }
      else {  // argumentCount< Function >() == 2
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
      for( Index globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx ); globalThreadIdx < elementsCount;
           globalThreadIdx += totalThreadsCount )
      {
         const Index segmentIdx = segmentIndexes[ globalThreadIdx / segmentSize + begin ];
         const Index localIdx = globalThreadIdx % segmentSize;
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }
   }
#endif
}

template< typename SegmentsConstView,
          typename ArrayView,
          typename Index,
          typename Function,
          ElementsOrganization Organization,
          int SegmentsPerBlock,
          int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesBlockMergeKernel_Ellpack( Index gridIdx,
                                                       SegmentsConstView segments,
                                                       ArrayView segmentIndexes,
                                                       const Index begin,
                                                       const Index end,
                                                       Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( segments.getSegmentSize() == 0 )
      return;
   const Index globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx );
   if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
      const Index localIdx = globalThreadIdx % segments.getSegmentSize();
      const Index segmentIdx_idx = begin + globalThreadIdx / segments.getSegmentSize();
      if( segmentIdx_idx >= end )
         return;
      TNL_ASSERT_GE( segmentIdx_idx, 0, "" );
      TNL_ASSERT_GE( localIdx, 0, "" );
      TNL_ASSERT_LT( localIdx, segments.getSegmentSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }
   else {  // ColumnMajorOrder
      const Index segmentsCount = ( end - begin );
      const Index localIdx = globalThreadIdx / segmentsCount;
      const Index segmentIdx_idx = begin + globalThreadIdx % segmentsCount;
      if( localIdx >= segments.getSegmentSize() )
         return;

      TNL_ASSERT_LT( segmentIdx_idx, end, "" );
      TNL_ASSERT_LT( localIdx, segments.getSegmentSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }

#endif
}

template< typename SegmentsView,
          typename Index,
          typename Condition,
          typename Function,
          ElementsOrganization Organization,
          int BlockSize = 256 >
__global__
void
forElementsIfKernel_Ellpack( const Index gridIdx,
                             const Index totalThreadsCount,
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

      if constexpr( argumentCount< Function >() == 3 ) {
         Index globalIdx = segmentIdx * segmentSize + laneIdx;
         for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            globalIdx += threadsPerSegment;
         }
      }
      else {  // argumentCount< Function >() == 2
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
      for( Index globalThreadIdx = Backend::getGlobalThreadIdx_x( gridIdx ); globalThreadIdx < elementsCount;
           globalThreadIdx += totalThreadsCount )
      {
         const Index segmentIdx = globalThreadIdx / segmentSize + begin;
         if( condition( segmentIdx ) ) {
            const Index localIdx = globalThreadIdx % segmentSize;
            const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( argumentCount< Function >() == 3 )
               function( segmentIdx, localIdx, globalIdx );
            else
               function( segmentIdx, globalIdx );
         }
      }
   }
#endif
}

template< typename SegmentsView,
          typename Index,
          typename Condition,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize,
          int SegmentsPerBlock,
          int BlockSize = 256 >
__global__
void
forElementsIfBlockMergeKernel_Ellpack( Index gridIdx,
                                       SegmentsView segments,
                                       const Index begin,
                                       const Index end,
                                       Condition condition,
                                       Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using InclusiveCudaScan =
      Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Inclusive, BlockSize, Plus, Index >;
   using ExclusiveCudaScan =
      Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using InclusiveScanStorage = typename InclusiveCudaScan::Storage;  // TODO: Storage should not depend on the scan type
   using ExclusiveScanStorage = typename ExclusiveCudaScan::Storage;

   __shared__ InclusiveScanStorage inclusive_scan_storage;
   //__shared__ ExclusiveScanStorage exclusive_scan_storage;
   __shared__ Index conditions[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];
   __shared__ Index segmentSize;

   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx = min( SegmentsPerBlock, end - begin - blockIdx.x * SegmentsPerBlock );
   Index conditionValue = 0;
   if( segmentIdx < end && threadIdx.x < SegmentsPerBlock ) {
      conditionValue = condition( segmentIdx );
   }
   __syncthreads();

   #ifdef USE_CUB1
   using BlockScan = cub::BlockScan< Index, 256 >;
   __shared__ typename BlockScan::TempStorage temp_storage;
   BlockScan( temp_storage ).InclusiveSum( conditionValue, conditionValue );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = conditionValue;
   #else  // USE_CUB
   const Index v1 = InclusiveCudaScan::scan( Plus{}, (Index) 0, conditionValue, threadIdx.x, inclusive_scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = v1;
   #endif
   __syncthreads();

   __shared__ Index activeSegmentsCount;
   if( threadIdx.x == 0 )
      activeSegmentsCount = conditions[ SegmentsPerBlock - 1 ];
   if( ( threadIdx.x == 0 && conditions[ 0 ] != 0 ) || conditions[ threadIdx.x ] != conditions[ threadIdx.x - 1 ] )
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
   __syncthreads();

   const Index last_idx = activeSegmentsCount * segments.getSegmentSize();
   TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_segment_indexes[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      if constexpr( Organization == RowMajorOrder ) {
         const Index local_segmentIdx = idx / segmentSize;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, activeSegmentsCount, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );

         const Index localIdx = idx % segmentSize;
         const Index segmentIdx = shared_segment_indexes[ local_segmentIdx ];
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }
      else {  // ColumnMajorOrder
         const Index localIdx = idx / activeSegmentsCount;
         const Index segmentIdx = shared_segment_indexes[ idx % activeSegmentsCount ];
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }

      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
