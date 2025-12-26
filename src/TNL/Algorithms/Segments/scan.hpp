// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/TypeTraits.h>
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

template< typename SegmentView, typename Fetch, typename Reduce, typename Write >
__cuda_callable__
void
inclusiveScanSegment( SegmentView& segment, Fetch&& fetch, Reduce&& reduce, Write&& write )
{
   using IndexType = typename SegmentView::IndexType;
   using ValueType = decltype( fetch( std::declval< IndexType >(), std::declval< IndexType >(), std::declval< IndexType >() ) );

   if( segment.getSize() == 0 )
      return;

   IndexType globalIdx = segment.getGlobalIndex( 0 );
   ValueType sum = detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segment.getSegmentIndex(), 0, globalIdx );
   write( globalIdx, sum );

   for( IndexType localIdx = 1; localIdx < segment.getSize(); localIdx++ ) {
      const IndexType globalIdx = segment.getGlobalIndex( localIdx );
      sum = reduce(
         sum, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segment.getSegmentIndex(), localIdx, globalIdx ) );
      write( globalIdx, sum );
   }
}

template< typename SegmentView, typename Fetch, typename Reduce, typename Write >
__cuda_callable__
void
exclusiveScanSegment( SegmentView& segment, Fetch&& fetch, Reduce&& reduce, Write&& write )
{
   using IndexType = typename SegmentView::IndexType;
   using ValueType = decltype( fetch( std::declval< IndexType >(), std::declval< IndexType >(), std::declval< IndexType >() ) );

   if( segment.getSize() == 0 )
      return;

   ValueType sum = reduce.template getIdentity< ValueType >();

   for( IndexType localIdx = 0; localIdx < segment.getSize(); localIdx++ ) {
      const IndexType globalIdx = segment.getGlobalIndex( localIdx );
      const ValueType previousValue =
         detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segment.getSegmentIndex(), localIdx, globalIdx );
      write( globalIdx, sum );
      sum = reduce( sum, previousValue );
   }
}

template< typename Segments, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanAllSegments( const Segments& segments,
                          Fetch&& fetch,
                          Reduce&& reduce,
                          Write&& write,
                          LaunchConfiguration launchConfig )
{
   inclusiveScanSegments( segments,
                          0,
                          segments.getSegmentsCount(),
                          std::forward< Fetch >( fetch ),
                          std::forward< Reduce >( reduce ),
                          std::forward< Write >( write ),
                          launchConfig );
}

template< typename Segments, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanAllSegments( const Segments& segments,
                          Fetch&& fetch,
                          Reduce&& reduce,
                          Write&& write,
                          LaunchConfiguration launchConfig )
{
   exclusiveScanSegments( segments,
                          0,
                          segments.getSegmentsCount(),
                          std::forward< Fetch >( fetch ),
                          std::forward< Reduce >( reduce ),
                          std::forward< Write >( write ),
                          launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduce, typename Write, typename T >
void
inclusiveScanSegments( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegments(
      segments,
      begin,
      end,
      [ = ] __cuda_callable__( SegmentView & segment ) mutable
      {
         inclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduce, typename Write, typename T >
void
exclusiveScanSegments( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegments(
      segments,
      begin,
      end,
      [ = ] __cuda_callable__( SegmentView & segment ) mutable
      {
         exclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

template< typename Segments, typename Array, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegments(
      segments,
      segmentIndexes,
      [ = ] __cuda_callable__( SegmentView & segment ) mutable
      {
         inclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

template< typename Segments, typename Array, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegments(
      segments,
      segmentIndexes,
      [ = ] __cuda_callable__( SegmentView & segment ) mutable
      {
         exclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanAllSegmentsIf( const Segments& segments,
                            Condition&& condition,
                            Fetch&& fetch,
                            Reduce&& reduce,
                            Write&& write,
                            LaunchConfiguration launchConfig )
{
   inclusiveScanSegmentsIf( segments,
                            0,
                            segments.getSegmentsCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            std::forward< Reduce >( reduce ),
                            std::forward< Write >( write ),
                            launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanAllSegmentsIf( const Segments& segments,
                            Condition&& condition,
                            Fetch&& fetch,
                            Reduce&& reduce,
                            Write&& write,
                            LaunchConfiguration launchConfig )
{
   exclusiveScanSegmentsIf( segments,
                            0,
                            segments.getSegmentsCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            std::forward< Reduce >( reduce ),
                            std::forward< Write >( write ),
                            launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T >
void
inclusiveScanSegmentsIf( const Segments& segments,
                         IndexBegin begin,
                         IndexEnd end,
                         Condition&& condition,
                         Fetch&& fetch,
                         Reduce&& reduce,
                         Write&& write,
                         LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegmentsIf(
      segments,
      begin,
      end,
      condition,
      [ = ] __cuda_callable__( const SegmentView& segment ) mutable
      {
         inclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T >
void
exclusiveScanSegmentsIf( const Segments& segments,
                         IndexBegin begin,
                         IndexEnd end,
                         Condition&& condition,
                         Fetch&& fetch,
                         Reduce&& reduce,
                         Write&& write,
                         LaunchConfiguration launchConfig )
{
   using SegmentView = typename Segments::SegmentViewType;

   forSegmentsIf(
      segments,
      begin,
      end,
      condition,
      [ = ] __cuda_callable__( const SegmentView& segment ) mutable
      {
         exclusiveScanSegment(
            segment, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), std::forward< Write >( write ) );
      },
      launchConfig );
}

}  // namespace TNL::Algorithms::Segments
