// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

#include "SlicedEllpackKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
template< typename Segments >
void
SlicedEllpackKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
SlicedEllpackKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
SlicedEllpackKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
SlicedEllpackKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
SlicedEllpackKernel< Index, Device >::getKernelType()
{
   return "SlicedEllpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
SlicedEllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                      Index begin,
                                                      Index end,
                                                      Fetch& fetch,
                                                      const Reduction& reduction,
                                                      ResultKeeper& keeper,
                                                      const Value& identity )
{
   using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
   const auto sliceOffsets = segments.getSliceOffsetsView();

   auto l = [ sliceOffsets, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
               const IndexType segmentIdx ) mutable
   {
      const IndexType sliceIdx = segmentIdx / SegmentsView::getSliceSize();
      const IndexType segmentInSliceIdx = segmentIdx % SegmentsView::getSliceSize();
      ReturnType aux = identity;
      IndexType localIdx = 0;

      if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
         const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
         const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;

         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            aux = reduction(
               aux,
               Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
      }
      else {
         (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler optimize it out...
         const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets[ sliceIdx + 1 ];

         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsView::getSliceSize() )
            aux = reduction(
               aux,
               Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
      }
      keeper( segmentIdx, aux );
   };

   Algorithms::parallelFor< Device >( begin, end, l );
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
SlicedEllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                         Fetch& fetch,
                                                         const Reduction& reduction,
                                                         ResultKeeper& keeper,
                                                         const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
