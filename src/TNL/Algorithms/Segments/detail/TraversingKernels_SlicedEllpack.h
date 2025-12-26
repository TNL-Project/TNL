// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

//#define USE_CUB
#ifdef __CUDACC__
   #include <cub/cub.cuh>
#endif

namespace TNL::Algorithms::Segments::detail {

template< typename SegmentsConstView,
          typename Index,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize,
          int BlockSize = 256 >
__global__
void
forElementsBlockMergeKernel_SlicedEllpack( const Index gridIdx,
                                           const SegmentsConstView segments,
                                           Index begin,
                                           Index end,
                                           Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index shared_offsets[ BlockSize / SliceSize + 1 ];
   __shared__ Index shared_segments_size[ BlockSize / SliceSize ];

   __shared__ Index firstSliceIdx,  // the slice corresponding to the segment with index `begin`
      lastSliceIdx,                 // the slice corresponding to the segment with index `end - 1`
      firstSegmentIdx,              // the index of the first segment in the first slice
      lastSegmentIdx,               // the index of the last segment in the last slice (increased by one)
      firstSegmentInBlock,          // the index of the first segment being processed in this block of threads
      firstSliceInBlockIdx,         // the index of the first slice being processed in this block of threads
      firstSegmentInBlockOffset,    // the offset of the first segment in this block of threads
      slicesInBlockCount;           // the number of slices being processed in this block of threads

   if( threadIdx.x == 0 ) {
      firstSliceIdx = begin / SliceSize;
      lastSliceIdx = end / SliceSize;
      firstSegmentIdx = firstSliceIdx * SliceSize;
      lastSegmentIdx = lastSliceIdx * SliceSize;
   }
   __syncthreads();

   const Index segmentIdx = firstSegmentIdx + Backend::getGlobalThreadIdx_x( gridIdx );
   const Index segmentInSliceIdx = segmentIdx % SliceSize;
   const Index sliceIdx = segmentIdx / SliceSize;
   if( threadIdx.x == 0 ) {
      firstSegmentInBlock = segmentIdx;
      firstSliceInBlockIdx = firstSegmentInBlock / SliceSize;
      firstSegmentInBlockOffset = segments.getGlobalIndex( firstSegmentInBlock, 0 );  // segmentIdx
      slicesInBlockCount = min( lastSliceIdx - firstSliceInBlockIdx + 1, BlockSize / SliceSize );
   }
   __syncthreads();

   if( segmentInSliceIdx == 0 && segmentIdx < end ) {
      shared_offsets[ sliceIdx - firstSliceInBlockIdx ] = segments.getGlobalIndex( segmentIdx, 0 ) - firstSegmentInBlockOffset;
      shared_segments_size[ sliceIdx - firstSliceInBlockIdx ] = segments.getSliceSegmentSizesView()[ sliceIdx ];
   }
   __syncthreads();
   if( threadIdx.x == 0 && segmentIdx < end ) {
      TNL_ASSERT_LT( slicesInBlockCount + firstSliceInBlockIdx - 1, segments.getSliceSegmentSizesView().getSize(), "" );
      shared_offsets[ slicesInBlockCount ] =
         shared_offsets[ slicesInBlockCount - 1 ]
         + SliceSize * segments.getSliceSegmentSizesView()[ slicesInBlockCount + firstSliceInBlockIdx - 1 ];
   }
   __syncthreads();

   const Index last_idx = shared_offsets[ slicesInBlockCount ];

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_sliceIdx ] = Algorithms::findUpperBound( shared_offsets, slicesInBlockCount + 1, idx );
      if( found ) {
         local_sliceIdx--;

         const Index inSliceIdx = idx - shared_offsets[ local_sliceIdx ];

         const Index local_segmentIdx = local_sliceIdx * SliceSize + inSliceIdx / shared_segments_size[ local_sliceIdx ];

         const Index globalIdx = firstSegmentInBlockOffset + idx;
         Index localIdx, currentSegmentIdx;
         if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
            localIdx = inSliceIdx % shared_segments_size[ local_sliceIdx ];
            currentSegmentIdx = firstSegmentInBlock + local_segmentIdx;
         }
         else {  // ColumnMajorOrder
            localIdx = inSliceIdx / SliceSize;
            currentSegmentIdx = firstSegmentInBlock + local_sliceIdx * SliceSize + inSliceIdx % SliceSize;
         }

         if( currentSegmentIdx >= begin && currentSegmentIdx < end ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( currentSegmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );

            if constexpr( argumentCount< Function >() == 3 )
               function( currentSegmentIdx, localIdx, globalIdx );
            else
               function( currentSegmentIdx, globalIdx );
         }
      }
      idx += BlockSize;
   }

#endif
}

template< typename SegmentsConstView, typename Index, typename Function, ElementsOrganization Organization, int SliceSize >
__global__
void
forElementsKernel_SlicedEllpack( const Index gridIdx,
                                 const Index threadsPerSegment,
                                 const SegmentsConstView segments,
                                 Index begin,
                                 Index end,
                                 Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx >= end )
         return;

      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      const Index sliceIdx = segmentIdx / SliceSize;
      const Index segmentInSliceIdx = segmentIdx % SliceSize;
      const Index segmentSize = segments.getSliceSegmentSizesView()[ sliceIdx ];

      const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + segmentInSliceIdx * segmentSize;
      const Index endIdx = beginIdx + segmentSize;
      TNL_ASSERT_EQ( beginIdx, segments.getGlobalIndex( segmentIdx, 0 ), "" );

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
         TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
         TNL_ASSERT_LT( globalIdx, endIdx, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         localIdx += threadsPerSegment;
      }
   }
   else {  // ColumnMajorOrder
      const Index firstSliceIdx = begin / SliceSize;
      const Index sliceIdx = firstSliceIdx + Backend::getGlobalThreadIdx_x( gridIdx ) / ( SliceSize * threadsPerSegment );
      const Index inSliceIdx = Backend::getGlobalThreadIdx_x( gridIdx ) % ( SliceSize * threadsPerSegment );
      const Index inSliceSegmentIdx = inSliceIdx % SliceSize;
      const Index segmentIdx = sliceIdx * SliceSize + inSliceSegmentIdx;

      if( segmentIdx < begin || segmentIdx >= end )
         return;

      TNL_ASSERT_LT( sliceIdx, segments.getSliceSegmentSizesView().getSize(), "" );
      TNL_ASSERT_LT( inSliceSegmentIdx, SliceSize, "" );
      TNL_ASSERT_LT( segmentIdx, segments.getSegmentsCount(), "" );
      Index localIdx = inSliceIdx / SliceSize;

      const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceSegmentIdx + localIdx * SliceSize;
      const Index endIdx = segments.getSliceOffsetsView()[ sliceIdx + 1 ];
      TNL_ASSERT_LE( endIdx, segments.getStorageSize(), "" );  // equality is for the case when both values equal 0

      const Index step = threadsPerSegment * SliceSize;
      for( Index globalIdx = beginIdx; globalIdx < endIdx; globalIdx += step ) {
         TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
         TNL_ASSERT_LT( globalIdx, endIdx, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         localIdx += threadsPerSegment;
      }
   }
#endif
}

template< typename SegmentsConstView,
          typename ArrayView,
          typename Index,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize >
__global__
void
forElementsWithSegmentIndexesKernel_SlicedEllpack( const Index gridIdx,
                                                   const Index threadsPerSegment,
                                                   const SegmentsConstView segments,
                                                   const ArrayView segmentIndexes,
                                                   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index idx = Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( idx >= segmentIndexes.getSize() )
      return;
   TNL_ASSERT_GE( idx, 0, "" );
   TNL_ASSERT_LT( idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ idx ];
   TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
   TNL_ASSERT_LT( segmentIdx, segments.getSegmentsCount(), "Wrong index segment index - larger that the number of indexes." );
   const Index sliceIdx = segmentIdx / SliceSize;
   const Index inSliceOffset = segmentIdx % SliceSize;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   const Index segmentSize = segments.getSliceSegmentSizesView()[ sliceIdx ];

   if constexpr( Organization == RowMajorOrder ) {
      Index globalIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceOffset * segmentSize + laneIdx;
      for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         globalIdx += threadsPerSegment;
      }
   }
   else {
      Index globalIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceOffset + laneIdx * SliceSize;
      for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         globalIdx += threadsPerSegment * SliceSize;
      }
   }
#endif
}

template< typename SegmentsConstView,
          typename ArrayView,
          typename Index,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize,
          int SegmentsPerBlock,
          int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesBlockMergeKernel_SlicedEllpack( const Index gridIdx,
                                                             const SegmentsConstView segments,
                                                             const ArrayView segmentIndexes,
                                                             Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using CudaScan = Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using ScanStorage = typename CudaScan::Storage;

   __shared__ ScanStorage scan_storage;
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx_ptr = Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx = min( SegmentsPerBlock, segmentIndexes.getSize() - blockIdx.x * SegmentsPerBlock );
   if( segmentIdx_ptr < segmentIndexes.getSize() && threadIdx.x < SegmentsPerBlock ) {
      TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      shared_segment_indexes[ threadIdx.x ] = seg_idx;
      TNL_ASSERT_GE( shared_segment_indexes[ threadIdx.x ], 0, "" );
      TNL_ASSERT_LT( shared_segment_indexes[ threadIdx.x ], segments.getSegmentsCount(), "" );
      shared_global_offsets[ threadIdx.x ] = segments.getGlobalIndex( seg_idx, 0 );
   }

   #ifdef USE_CUB
   using BlockScan = cub::BlockScan< Index, 256 >;
   __shared__ typename BlockScan::TempStorage temp_storage;
   Index value = 0;
   if( segmentIdx_ptr < end && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      value = segments.getSegmentSize( seg_idx );
   }
   BlockScan( temp_storage ).ExclusiveSum( value, value );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = value;
   #else  // USE_CUB
   Index value = 0;
   if( segmentIdx_ptr < segmentIndexes.getSize() && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT(
         seg_idx, segments.getSegmentsCount(), "Wrong index of segment index - larger that the number of indexes." );
      value = segments.getSegmentSize( seg_idx );
   }
   const Index v = CudaScan::scan( Plus{}, (Index) 0, value, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v;
   #endif

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 ) {
         TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
         const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT(
            seg_idx, segments.getSegmentsCount(), "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + segments.getSegmentSize( seg_idx );
      }
   __syncthreads();

   const Index last_idx = shared_offsets[ last_local_segment_idx ];
   TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_offsets[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, last_local_segment_idx + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, last_local_segment_idx, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], segments.getSegmentsCount(), "" );

         if constexpr( Organization == RowMajorOrder ) {
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( argumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
         else {  // ColumnMajorOrder
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx * SliceSize;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( argumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
      }
      idx += BlockSize;
   }
#endif
}

template< typename SegmentsConstView,
          typename Index,
          typename Condition,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize,
          int BlockSize = 256 >
__global__
void
forElementsIfKernel_SlicedEllpack( const Index gridIdx,
                                   const Index threadsPerSegment,
                                   const SegmentsConstView segments,
                                   Index begin,
                                   Index end,
                                   Condition condition,
                                   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   const Index segmentSize = segments.getSegmentSize( segmentIdx );

   if constexpr( Organization == RowMajorOrder ) {
      Index globalIdx = segments.getGlobalIndex( segmentIdx, 0 );
      const Index endIdx = globalIdx + segmentSize;
      globalIdx += laneIdx;
      if constexpr( argumentCount< Function >() == 3 ) {
         Index localIdx = laneIdx;
         for( ; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            localIdx += threadsPerSegment;
         }
      }
      else {  // argumentCount< Function >() == 2
         for( ; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
   else {  // Organization == ColumnMajorOrder
      Index globalIdx = segments.getGlobalIndex( segmentIdx, laneIdx );
      const Index endIdx = segments.getGlobalIndex( segmentIdx, segmentSize );
      if constexpr( argumentCount< Function >() == 3 ) {
         Index localIdx = laneIdx;
         for( ; globalIdx < endIdx; globalIdx += SliceSize * threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            localIdx += threadsPerSegment;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx += SliceSize * threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
#endif
}

template< typename SegmentsConstView,
          typename Index,
          typename Condition,
          typename Function,
          ElementsOrganization Organization,
          int SliceSize,
          int SegmentsPerBlock,
          int BlockSize = 256 >
__global__
void
forElementsIfBlockMergeKernel_SlicedEllpack( const Index gridIdx,
                                             const SegmentsConstView segments,
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
   __shared__ ExclusiveScanStorage exclusive_scan_storage;
   __shared__ Index conditions[ SegmentsPerBlock ];
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx = min( SegmentsPerBlock, end - begin - blockIdx.x * SegmentsPerBlock );
   Index conditionValue = 0;
   if( segmentIdx < end && threadIdx.x < SegmentsPerBlock ) {
      conditionValue = condition( segmentIdx );
      shared_offsets[ threadIdx.x ] = 0;
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
   if( ( threadIdx.x == 0 && conditions[ 0 ] != 0 ) || conditions[ threadIdx.x ] != conditions[ threadIdx.x - 1 ] ) {
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
      shared_global_offsets[ conditions[ threadIdx.x ] - 1 ] =
         segments.getGlobalIndex( segmentIdx, 0 );  // TODO: get this using sliceIdx
      shared_offsets[ conditions[ threadIdx.x ] - 1 ] = segments.getSegmentSize( segmentIdx );
   }
   __syncthreads();

   Index segmentSize = shared_offsets[ threadIdx.x ];
   #ifdef USE_CUB
   using BlockScan = cub::BlockScan< Index, 256 >;
   BlockScan( temp_storage ).ExclusiveSum( segmentSize, segmentSize );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = segmentSize;
   #else  // USE_CUB
   const Index v2 = ExclusiveCudaScan::scan( Plus{}, (Index) 0, segmentSize, threadIdx.x, exclusive_scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v2;
   #endif

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 && activeSegmentsCount == BlockSize ) {
         const Index seg_idx = shared_segment_indexes[ threadIdx.x ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT(
            seg_idx, segments.getSegmentsCount() - 1, "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + segments.getSegmentSize( seg_idx );
      }
   __syncthreads();

   const Index last_idx = shared_offsets[ activeSegmentsCount ];
   TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_segment_indexes[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, activeSegmentsCount + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, activeSegmentsCount, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], segments.getSegmentsCount(), "" );

         if constexpr( Organization == RowMajorOrder ) {
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( argumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
         else {  // ColumnMajorOrder
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx * SliceSize;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( argumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
      }
      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
