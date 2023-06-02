// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

#include "detail/FetchLambdaAdapter.h"
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
TNL::String
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
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
   const auto sliceOffsets = segments.getSliceOffsetsView();
   if( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsView::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsView::getSliceSize();
         const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
         const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         ReturnType aux = identity;
         IndexType localIdx = 0;
         bool compute = true;
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            aux = reduction(
               aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
   else {
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsView::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsView::getSliceSize();
         // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets[ sliceIdx + 1 ];
         ReturnType aux = identity;
         IndexType localIdx = 0;
         bool compute = true;
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsView::getSliceSize() )
            aux = reduction(
               aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
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
