// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "find.h"
#include "reduce.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename Condition, typename ResultStorer >
static void
findInAllSegments( const Segments& segments, Condition&& condition, ResultStorer&& storer, LaunchConfiguration launchConfig )
{
   findInSegments( segments,
                   (typename Segments::IndexType) 0,
                   segments.getSegmentsCount(),
                   std::forward< Condition >( condition ),
                   std::forward< ResultStorer >( storer ),
                   launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename ResultStorer, typename T >
static void
findInSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                ResultStorer&& storer,
                LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   auto store_ = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, bool found, bool emptySegment ) mutable
   {
      storer( segmentIdx, localIdx, found );
   };
   reduceSegmentsWithArgument(
      segments, begin, end, std::forward< Condition >( condition ), AnyWithArg{}, std::move( store_ ), launchConfig );
}

template< typename Segments, typename Array, typename Condition, typename ResultStorer, typename T >
static void
findInSegments( const Segments& segments,
                const Array& segmentIndexes,
                Condition&& condition,
                ResultStorer&& storer,
                LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   auto store_ = [ = ] __cuda_callable__(
                    IndexType segmentIdx_idx, IndexType segmentIdx, IndexType localIdx, bool found, bool emptySegment ) mutable
   {
      storer( segmentIdx_idx, segmentIdx, localIdx, found );
   };
   reduceSegmentsWithArgument(
      segments, segmentIndexes, std::forward< Condition >( condition ), AnyWithArg{}, std::move( store_ ), launchConfig );
}

template< typename Segments, typename SegmentCondition, typename Condition, typename ResultStorer >
static void
findInAllSegmentsIf( const Segments& segments,
                     SegmentCondition&& segmentCondition,
                     Condition&& condition,
                     ResultStorer&& storer,
                     LaunchConfiguration launchConfig )
{
   findInSegmentsIf( segments,
                     (typename Segments::IndexType) 0,
                     segments.getSegmentsCount(),
                     std::forward< SegmentCondition >( segmentCondition ),
                     std::forward< Condition >( condition ),
                     std::forward< ResultStorer >( storer ),
                     launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentCondition,
          typename Condition,
          typename ResultStorer,
          typename T >
static void
findInSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  SegmentCondition&& segmentCondition,
                  Condition&& condition,
                  ResultStorer&& storer,
                  LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   auto store_ = [ = ] __cuda_callable__(
                    IndexType segmentIdx_idx, IndexType segmentIdx, IndexType localIdx, bool found, bool emptySegment ) mutable
   {
      storer( segmentIdx, localIdx, found );
   };
   reduceSegmentsWithArgumentIf( segments,
                                 begin,
                                 end,
                                 std::forward< SegmentCondition >( segmentCondition ),
                                 std::forward< Condition >( condition ),
                                 AnyWithArg{},
                                 std::move( store_ ),
                                 launchConfig );
}

}  // namespace TNL::Algorithms::Segments
