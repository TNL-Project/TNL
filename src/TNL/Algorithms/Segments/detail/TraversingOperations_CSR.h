// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "TraversingKernels_CSR.h"
#include "TraversingOperationsBaseline.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct TraversingOperations< CSRView< Device, Index > > : public TraversingOperationsBaseline< CSRView< Device, Index > >
{
   using ViewType = CSRView< Device, Index >;
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
      const auto offsetsView = segments.getOffsets();
      // TODO: if constexpr could be just inside the lambda function l when nvcc allolws it
      if constexpr( argumentCount< Function >() == 3 ) {
         auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            IndexType localIdx( 0 );
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               function( segmentIdx, localIdx++, globalIdx );
         };
         // TODO: Add launch config
         Algorithms::parallelFor< Device >( begin, end, l );
      }
      else {
         auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               function( segmentIdx, globalIdx );
         };
         // TODO: Add launch config
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() != ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
         else {
            const Index segmentsCount = end - begin;
            std::size_t threadsCount = segmentsCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment )
               threadsCount = segmentsCount * Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMergedSegments ) {
               threadsCount = segmentsCount / launchConfig.getThreadsPerSegmentCount();
               launchConfig.blockSize.x = 256;
            }

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel = forElementsKernel_CSR< ConstOffsetsView, IndexType, Function >;
                  Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments.getOffsets(), begin, end, function );
               }
               else {  // This mapping is currently the default one
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        constexpr auto kernel = forElementsBlockMergeKernel_CSR< ConstOffsetsView, IndexType, Function >;
                        Backend::launchKernelAsync(
                           kernel, launchConfig, gridIdx, segments.getOffsets(), begin, end, function );
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
                          LaunchConfiguration launchConfig )
   {
      q const auto offsetsView = segments.getOffsets();
      auto segmentIndexesView = segmentIndexes.getConstView();
      // TODO: if constexpr could be just inside the lambda function l when nvcc allolws it
      if constexpr( argumentCount< Function >() == 3 ) {
         auto l = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            TNL_ASSERT_LT( idx, segmentIndexesView.getSize(), "" );
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
            TNL_ASSERT_LT(
               segmentIdx, offsetsView.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            IndexType localIdx( 0 );
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
               TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
               function( segmentIdx, localIdx++, globalIdx );
            }
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
      else {  // argumentCount< Function >() == 2
         auto l = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            TNL_ASSERT_LT( idx, segmentIndexesView.getSize(), "" );
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
            TNL_ASSERT_LT(
               segmentIdx, offsetsView.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
               TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
               function( segmentIdx, globalIdx );
            }
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function function,  // TODO: Function&& function does not work here
                LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;

      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, segmentIndexes, begin, end, std::forward< Function >( function ), launchConfig );
         else {
            auto segmentIndexesView = segmentIndexes.getConstView();
            const Index segmentsCount = end - begin;
            std::size_t threadsCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment )
               threadsCount = segmentsCount * Backend::getWarpSize();
            else {  // This mapping is currently the default one
               launchConfig.blockSize.x = 256;
               threadsCount = segmentsCount * launchConfig.getThreadsPerSegmentCount();
            }

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel = detail::forElementsWithSegmentIndexesKernel_CSR< ConstOffsetsView,
                                                                                           typename Array::ConstViewType,
                                                                                           IndexType,
                                                                                           Function >;
                  Backend::launchKernelAsync(
                     kernel, launchConfig, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );
               }
               else {  // This mapping is currently the default one
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        {
                           constexpr int SegmentsPerBlock = 256;
                           constexpr auto kernel =
                              detail::forElementsWithSegmentIndexesBlockMergeKernel_CSR< ConstOffsetsView,
                                                                                         typename Array::ConstViewType,
                                                                                         IndexType,
                                                                                         Function,
                                                                                         SegmentsPerBlock >;
                           Backend::launchKernelAsync(
                              kernel, launchConfig, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );
                           break;
                        }
                     case 2:
                        {
                           constexpr int SegmentsPerBlock = 128;
                           constexpr auto kernel =
                              detail::forElementsWithSegmentIndexesBlockMergeKernel_CSR< ConstOffsetsView,
                                                                                         typename Array::ConstViewType,
                                                                                         IndexType,
                                                                                         Function,
                                                                                         SegmentsPerBlock >;
                           Backend::launchKernelAsync(
                              kernel, launchConfig, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );
                           break;
                        }
                     case 4:
                        {
                           constexpr int SegmentsPerBlock = 64;
                           constexpr auto kernel =
                              detail::forElementsWithSegmentIndexesBlockMergeKernel_CSR< ConstOffsetsView,
                                                                                         typename Array::ConstViewType,
                                                                                         IndexType,
                                                                                         Function,
                                                                                         SegmentsPerBlock >;
                           Backend::launchKernelAsync(
                              kernel, launchConfig, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );
                           break;
                        }
                     case 8:
                        {
                           constexpr int SegmentsPerBlock = 32;
                           constexpr auto kernel =
                              detail::forElementsWithSegmentIndexesBlockMergeKernel_CSR< ConstOffsetsView,
                                                                                         typename Array::ConstViewType,
                                                                                         IndexType,
                                                                                         Function,
                                                                                         SegmentsPerBlock >;
                           Backend::launchKernelAsync(
                              kernel, launchConfig, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );
                           break;
                        }
                  }
               }
            }
            Backend::streamSynchronize( launchConfig.stream );
         }
      }
      else
         forElementsSequential( segments, segmentIndexes, begin, end, std::forward< Function >( function ), launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSequential( const ConstViewType& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Condition condition,
                            Function function,
                            LaunchConfiguration launchConfig )
   {
      const auto offsetsView = segments.getOffsets();
      auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
      {
         const IndexType begin = offsetsView[ segmentIdx ];
         const IndexType end = offsetsView[ segmentIdx + 1 ];

         if( condition( segmentIdx ) ) {
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {  // argumentCount< Function >() == 2
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
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
                  constexpr auto kernel = forElementsIfKernel_CSR< ConstOffsetsView, IndexType, Condition, Function >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getOffsets(), begin, end, condition, function );
               }
               else {  // BlockMerge mapping - this mapping is currently the default one
                  constexpr auto kernel =
                     forElementsIfBlockMergeKernel_CSR< ConstOffsetsView, IndexType, Condition, Function, 256, 256 >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getOffsets(), begin, end, condition, function );
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
