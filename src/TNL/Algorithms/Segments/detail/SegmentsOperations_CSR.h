// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "CSRKernels.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct SegmentsOperations< CSRView< Device, Index > >
{
   using ViewType = CSRView< Device, Index >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = Index;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          IndexBegin begin,
                          IndexEnd end,
                          const LaunchConfiguration& launchConfig,
                          Function&& function )
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
                LaunchConfiguration launchConfig,
                Function&& function )
   {
      if( end <= begin )
         return;
      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() != ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, begin, end, launchConfig, std::forward< Function >( function ) );
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
                  constexpr auto kernel = forElementsKernel< ConstOffsetsView, IndexType, Function >;
                  Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments.getOffsets(), begin, end, function );
               }
               else {  // This mapping is currently the default one
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        constexpr auto kernel = forElementsBlockMergeKernel< ConstOffsetsView, IndexType, Function >;
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
         forElementsSequential( segments, begin, end, launchConfig, std::forward< Function >( function ) );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                LaunchConfiguration launchConfig,
                Function&& function )
   {
      return forElements( segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          const Array& segmentIndexes,
                          IndexBegin begin,
                          IndexEnd end,
                          LaunchConfiguration launchConfig,
                          Function&& function )
   {
      const auto offsetsView = segments.getOffsets();
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
                LaunchConfiguration launchConfig,
                Function&& function )
   {
      if( end <= begin )
         return;
      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment )
            forElementsSequential( segments, segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
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
                  constexpr auto kernel = detail::forElementsWithSegmentIndexesKernel< ConstOffsetsView,
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
                              detail::forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
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
                              detail::forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
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
                              detail::forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
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
                              detail::forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
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
         forElementsSequential( segments, segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                LaunchConfiguration launchConfig,
                Function&& function )
   {
      return forElements(
         segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  LaunchConfiguration launchConfig,
                  Condition condition,
                  Function function )
   {
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         if( end <= begin )
            return;

         const Index warpsCount = end - begin;
         const std::size_t threadsCount = warpsCount * Backend::getWarpSize();
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         dim3 blocksCount;
         dim3 gridsCount;
         Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
         for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
            Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
            constexpr auto kernel = forElementsIfKernel< ConstOffsetsView, IndexType, Condition, Function >;
            Backend::launchKernelAsync(
               kernel, launch_config, gridIdx, segments.getOffsets(), begin, end, condition, function );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
      else {
         const auto offsetsView = segments.getOffsets();
         auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            IndexType localIdx( 0 );
            if( condition( segmentIdx ) )
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  LaunchConfiguration launchConfig,
                  Condition condition,
                  Function function )
   {
      forElementsIf( segments.getConstView(),
                     begin,
                     end,
                     launchConfig,
                     std::forward< Condition >( condition ),
                     std::forward< Function >( function ) );
   }
};

template< typename Device, typename Index, typename IndexAllocator >
struct SegmentsOperations< CSR< Device, Index, IndexAllocator > >
{
   using SegmentsType = CSR< Device, Index, IndexAllocator >;
   using ViewType = typename SegmentsType::ViewType;
   using ConstViewType = typename SegmentsType::ViewType;
   using DeviceType = Device;
   using IndexType = Index;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const SegmentsType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition&& condition,
                  Function&& function )
   {
      SegmentsOperations< ViewType >::forElementsIf( segments.getConstView(),
                                                     begin,
                                                     end,
                                                     launchConfig,
                                                     std::forward< Condition >( condition ),
                                                     std::forward< Function >( function ) );
   }
};
}  //namespace TNL::Algorithms::Segments::detail
