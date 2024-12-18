// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct SegmentsOperations< SlicedEllpackView< Device, Index, Organization, SliceSize > >
{
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = Index;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
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
      else {
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
   forElements( const ViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      return forElements( segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
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
   forElements( const ViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
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
                  const LaunchConfiguration& launchConfig,
                  Condition condition,
                  Function function )
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
   forElementsIf( const ViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
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

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int SliceSize >
struct SegmentsOperations< SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize > >
{
   using SegmentsType = SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >;
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
