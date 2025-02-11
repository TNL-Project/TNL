// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct ReducingOperations< SlicedEllpackView< Device, Index, Organization, SliceSize > >
{
   using SegmentsViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;

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
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();

      auto l = [ sliceOffsets, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType aux = identity;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize() )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         keeper( segmentIdx, aux );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     IndexBegin begin,
                                     IndexEnd end,
                                     Fetch&& fetch,
                                     Reduction&& reduction,
                                     ResultKeeper&& keeper,
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ sliceOffsets, segmentIndexes_view, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx_idx ) mutable
      {
         TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType aux = identity;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize() )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         keeper( segmentIdx_idx, segmentIdx, aux );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();

      auto l = [ sliceOffsets, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType result = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize(), localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         keeper( segmentIdx, result, argument );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
