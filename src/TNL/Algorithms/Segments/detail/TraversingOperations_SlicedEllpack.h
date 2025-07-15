// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "TraversingKernels_SlicedEllpack.h"
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct TraversingOperations< SlicedEllpackView< Device, Index, Organization, SliceSize > >
: public TraversingOperationsBase< SlicedEllpackView< Device, Index, Organization, SliceSize > >
{
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          IndexBegin begin,
                          IndexEnd end,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      if constexpr( Organization == RowMajorOrder ) {
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {                                                 // ColumnMajorOrder
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function function,  // TODO: Function&& does not work here - why???
                LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;

      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
         else {
            std::size_t threadsCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
               const Index segmentsCount = end - begin;
               const IndexType slicesCount = roundUpDivision( segmentsCount, SliceSize );
               threadsCount = slicesCount * SliceSize * Backend::getWarpSize();
            }
            else  // launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMergedSegments
            {
               const IndexType firstSegmentIdx = ( begin / SliceSize ) * SliceSize;
               const IndexType lastSegmentIdx = roundUpDivision( end, SliceSize ) * SliceSize;
               const IndexType segmentsCount = lastSegmentIdx - firstSegmentIdx;
               threadsCount = segmentsCount / launchConfig.getThreadsPerSegmentCount();
               launchConfig.blockSize.x = 256;
            }

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel =
                     forElementsKernel_SlicedEllpack< ConstViewType, IndexType, Function, Organization, SliceSize >;
                  Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, begin, end, function );
               }
               else {  // This mapping is currently the default one
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        constexpr auto kernel = forElementsBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                           IndexType,
                                                                                           Function,
                                                                                           Organization,
                                                                                           SliceSize,
                                                                                           256 >;
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, begin, end, function );
                        break;
                  }
               }
            }
            Backend::streamSynchronize( launchConfig.stream );
         }
      }
      else {
         forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
      }
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          const Array& segmentIndexes,
                          IndexBegin begin,
                          IndexEnd end,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      auto segmentIndexes_view = segmentIndexes.getConstView();
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      if constexpr( Organization == RowMajorOrder ) {
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexes_view[ idx ];
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexes_view[ idx ];
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexes_view[ idx ];
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexes_view[ idx ];
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function function,  // TODO: Function&& does not work here - why???
                LaunchConfiguration launchConfig )
   {
      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, segmentIndexes, begin, end, std::forward< Function >( function ), launchConfig );
         else {
            auto segmentIndexesView = segmentIndexes.getConstView();
            std::size_t threadsCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
               const Index segmentsCount = end - begin;
               threadsCount = segmentsCount * Backend::getWarpSize();
            }
            else  // launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMergedSegments
            {
               const Index segmentsCount = end - begin;
               threadsCount = segmentsCount;  // * Backend::getWarpSize();
               launchConfig.blockSize.x = 256;
            }

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel = forElementsWithSegmentIndexesKernel_SlicedEllpack< ConstViewType,
                                                                                             typename Array::ConstViewType,
                                                                                             IndexType,
                                                                                             Function,
                                                                                             Organization,
                                                                                             SliceSize >;
                  Backend::launchKernelAsync(
                     kernel, launchConfig, gridIdx, segments, segmentIndexesView, begin, end, function );
               }
               else {  // This mapping is currently the default one
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        constexpr auto kernel =
                           forElementsWithSegmentIndexesBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                        typename Array::ConstViewType,
                                                                                        IndexType,
                                                                                        Function,
                                                                                        Organization,
                                                                                        SliceSize,
                                                                                        256,    // SegmentsPerBlock
                                                                                        256 >;  // BlockSize
                        Backend::launchKernelAsync(
                           kernel, launchConfig, gridIdx, segments, segmentIndexesView, begin, end, function );
                        break;
                  }
               }
            }
            Backend::streamSynchronize( launchConfig.stream );
         }
      }
      else {
         forElementsSequential( segments, segmentIndexes, begin, end, std::forward< Function >( function ), launchConfig );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSequential( const ConstViewType& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Condition condition,
                            Function function,
                            const LaunchConfiguration& launchConfig )
   {
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      if constexpr( Organization == RowMajorOrder ) {
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType sliceIdx = segmentIdx / SliceSize;
               const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
               // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
               const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
               const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition condition,
                  Function function,
                  LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         if( end <= begin )
            return;

         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsIfSequential( segments, begin, end, std::forward< Condition >( condition ), function, launchConfig );
         else {
            const Index warpsCount = end - begin;
            std::size_t threadsCount = warpsCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment )
               threadsCount = warpsCount * Backend::getWarpSize();
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel =
                     forElementsIfKernel_SlicedEllpack< ConstViewType, IndexType, Condition, Function, Organization, SliceSize >;
                  Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments, begin, end, condition, function );
               }
               else {  // BlockMerge mapping - this mapping is currently the default one
                  constexpr auto kernel = forElementsIfBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                       IndexType,
                                                                                       Condition,
                                                                                       Function,
                                                                                       Organization,
                                                                                       SliceSize,
                                                                                       256,
                                                                                       256 >;
                  Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments, begin, end, condition, function );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         forElementsIfSequential( segments, begin, end, std::forward< Condition >( condition ), function, launchConfig );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
