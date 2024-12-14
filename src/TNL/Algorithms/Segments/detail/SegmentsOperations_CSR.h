// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
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
   forElements( const ConstViewType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      if( end <= begin )
         return;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         const Index segmentsCount = end - begin;
         std::size_t threadsCount;
         if constexpr( argumentCount< Function >() == 2 )  // we use scan kernel
            threadsCount = segmentsCount;
         else
            threadsCount = segmentsCount * Backend::getWarpSize();
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         dim3 blocksCount;
         dim3 gridsCount;
         Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
         for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
            Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
            if constexpr( argumentCount< Function >() == 3 ) {
               constexpr auto kernel = forElementsKernel< ConstOffsetsView, IndexType, Function >;
               Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments.getOffsets(), begin, end, function );
            }
            else {
               constexpr auto kernel = forElementsBlockMergeKernel< ConstOffsetsView, IndexType, Function >;
               Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments.getOffsets(), begin, end, function );
            }
         }
         Backend::streamSynchronize( launch_config.stream );
      }
      else {
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
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      return forElements( segments.getConstView(), begin, end, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function )
   {
      if( end <= begin )
         return;
      auto segmentIndexesView = segmentIndexes.getConstView();
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         const Index segmentsCount = end - begin;
         std::size_t threadsCount;
         constexpr int ThreadsPerSegment = 16;
         constexpr int SegmentsPerBlock = 256 / ThreadsPerSegment;
         //threadsCount = segmentsCount * ThreadsPerSegment;  // for block merge kernel
         threadsCount = segmentsCount * Backend::getWarpSize();  // for vector kernel
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         dim3 blocksCount;
         dim3 gridsCount;
         Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
         for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
            Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

            constexpr auto kernel = detail::
               forElementsWithSegmentIndexesKernel< ConstOffsetsView, typename Array::ConstViewType, IndexType, Function >;
            Backend::launchKernelAsync(
               kernel, launch_config, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end, function );

            /*constexpr auto kernel = detail::forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
                                                                                   typename Array::ConstViewType,
                                                                                   IndexType,
                                                                                   Function,
                                                                                   SegmentsPerBlock >;
            Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments.getOffsets(), segmentIndexesView, begin, end,
            function
            );*/
         }
         Backend::streamSynchronize( launch_config.stream );
      }
      else {
         const auto offsetsView = segments.getOffsets();
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
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function&& function )
   {
      return forElements( segments.getConstView(), segmentIndexes, begin, end, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
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
   forElementsIf( const ViewType& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
   {
      forElementsIf(
         segments.getConstView(), begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ) );
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
   forElements( const SegmentsType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      SegmentsOperations< ViewType >::forElements( segments.getConstView(), begin, end, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), segmentIndexes, begin, end, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const SegmentsType& segments, IndexBegin begin, IndexEnd end, Condition&& condition, Function&& function )
   {
      SegmentsOperations< ViewType >::forElementsIf(
         segments.getConstView(), begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ) );
   }
};
}  //namespace TNL::Algorithms::Segments::detail