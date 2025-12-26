// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "find.h"
#include "reduce.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename ResultKeeper, typename T >
static void
findInSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   reduceSegmentsWithArgument( segments,
                               begin,
                               end,
                               std::forward< Condition >( condition ),
                               AnyWithArg{},
                               std::forward< ResultKeeper >( keeper ),
                               launchConfig );
}

template< typename Segments, typename Condition, typename ResultKeeper >
static void
findInAllSegments( const Segments& segments, Condition&& condition, ResultKeeper&& keeper, LaunchConfiguration launchConfig )
{
   findInSegments( segments,
                   (typename Segments::IndexType) 0,
                   segments.getSegmentsCount(),
                   std::forward< Condition >( condition ),
                   std::forward< ResultKeeper >( keeper ),
                   launchConfig );
}

template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename ResultKeeper,
          typename T >
static void
findInSegments( const Segments& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   reduceSegmentsWithArgument( segments,
                               segmentIndexes.getConstView( begin, end ),
                               std::forward< Condition >( condition ),
                               AnyWithArg{},
                               std::forward< ResultKeeper >( keeper ),
                               launchConfig );
}

template< typename Segments, typename Array, typename Condition, typename ResultKeeper, typename T >
static void
findInSegments( const Segments& segments,
                const Array& segmentIndexes,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   findInSegments( segments,
                   segmentIndexes,
                   (typename Segments::IndexType) 0,
                   segmentIndexes.getSize(),
                   std::forward< Condition >( condition ),
                   std::forward< ResultKeeper >( keeper ),
                   launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentCondition,
          typename Condition,
          typename ResultKeeper,
          typename T >
static void
findInSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  SegmentCondition&& segmentCondition,
                  Condition&& condition,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   auto keep_ =
      [ = ] __cuda_callable__(
         const IndexType segmentIdx_idx, const IndexType segmentIdx, const IndexType localIdx, const bool found ) mutable
   {
      keeper( segmentIdx, localIdx, found );
   };
   reduceSegmentsIfWithArgument( segments,
                                 begin,
                                 end,
                                 std::forward< SegmentCondition >( segmentCondition ),
                                 std::forward< Condition >( condition ),
                                 AnyWithArg{},
                                 keep_,
                                 launchConfig );
}

template< typename Segments, typename SegmentCondition, typename Condition, typename ResultKeeper >
static void
findInSegmentsIf( const Segments& segments,
                  SegmentCondition&& segmentCondition,
                  Condition&& condition,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig )
{
   findInSegmentsIf( segments,
                     (typename Segments::IndexType) 0,
                     segments.getSegmentsCount(),
                     std::forward< SegmentCondition >( segmentCondition ),
                     std::forward< Condition >( condition ),
                     std::forward< ResultKeeper >( keeper ),
                     launchConfig );
}

}  // namespace TNL::Algorithms::Segments
