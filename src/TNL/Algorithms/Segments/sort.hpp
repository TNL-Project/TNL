// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "sort.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

template< typename SegmentView, typename Fetch, typename Compare, typename Swap >
__cuda_callable__
void
segmentInsertionSort( SegmentView segment, Fetch&& fetch, Compare&& compare, Swap&& swap )
{
   using IndexType = typename SegmentView::IndexType;
   for( IndexType i = 1; i < segment.getSize(); i++ ) {
      for( IndexType j = i; j > 0; j-- ) {
         IndexType globalIdx1 = segment.getGlobalIndex( j - 1 );
         IndexType globalIdx2 = segment.getGlobalIndex( j );
         if( compare(
                detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segment.getSegmentIndex(), j - 1, globalIdx1 ),
                detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segment.getSegmentIndex(), j, globalIdx2 ) ) )
            break;
         swap( globalIdx1, globalIdx2 );
      }
   }
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Compare, typename Swap, typename T >
static void
sortSegments( const Segments& segments,
              IndexBegin begin,
              IndexEnd end,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;
   forSegments(
      segments,
      begin,
      end,
      [ = ] __cuda_callable__( const SegmentView& segment ) mutable
      {
         segmentInsertionSort( segment, fetch, compare, swap );
      },
      launchConfig );
}

template< typename Segments, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegments( const Segments& segments, Fetch&& fetch, Compare&& compare, Swap&& swap, LaunchConfiguration launchConfig )
{
   sortSegments( segments,
                 0,
                 segments.getSegmentsCount(),
                 std::forward< Fetch >( fetch ),
                 std::forward< Compare >( compare ),
                 std::forward< Swap >( swap ),
                 launchConfig );
}

template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T >
static void
sortSegments( const Segments& segments,
              const Array& segmentIndexes,
              IndexBegin begin,
              IndexEnd end,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;
   forSegments(
      segments,
      segmentIndexes.getConstView( begin, end ),
      [ = ] __cuda_callable__( const SegmentView segment ) mutable
      {
         segmentInsertionSort( segment, fetch, compare, swap );
      },
      launchConfig );
}

template< typename Segments, typename Array, typename Fetch, typename Compare, typename Swap, typename T >
static void
sortSegments( const Segments& segments,
              const Array& segmentIndexes,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig )
{
   sortSegments( segments,
                 segmentIndexes,
                 0,
                 segmentIndexes.getSize(),
                 std::forward< Fetch >( fetch ),
                 std::forward< Compare >( compare ),
                 std::forward< Swap >( swap ),
                 launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T >
static void
sortSegmentsIf( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                Fetch&& fetch,
                Compare&& compare,
                Swap&& swap,
                LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;
   forSegmentsIf(
      segments,
      begin,
      end,
      condition,
      [ = ] __cuda_callable__( const SegmentView segment ) mutable
      {
         segmentInsertionSort( segment, fetch, compare, swap );
      },
      launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegmentsIf( const Segments& segments,
                   Condition&& condition,
                   Fetch&& fetch,
                   Compare&& compare,
                   Swap&& swap,
                   LaunchConfiguration launchConfig )
{
   sortSegmentsIf( segments,
                   0,
                   segments.getSegmentsCount(),
                   std::forward< Condition >( condition ),
                   std::forward< Fetch >( fetch ),
                   std::forward< Compare >( compare ),
                   std::forward< Swap >( swap ),
                   launchConfig );
}

}  // namespace TNL::Algorithms::Segments
