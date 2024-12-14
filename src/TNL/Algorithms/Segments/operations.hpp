// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/SegmentsOperations.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename Index, typename Function >
void
forElements( const Segments& segments, Index begin, Index end, Function&& function )
{
   detail::SegmentsOperations< Segments >::forElements( segments, begin, end, std::forward< Function >( function ) );
}

template< typename Segments, typename Function >
void
forAllElements( const Segments& segments, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< Segments >::forElements(
      segments, (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function function )
{
   detail::SegmentsOperations< Segments >::forElements(
      segments, segmentIndexes, begin, end, std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, Function function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< Segments >::forElements(
      segments, segmentIndexes, (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

template< typename Segments, typename Index, typename Condition, typename Function >
void
forElementsIf( const Segments& segments, Index begin, Index end, Condition condition, Function function )
{
   detail::SegmentsOperations< Segments >::forElementsIf(
      segments, begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ) );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments& segments, Condition condition, Function function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< Segments >::forElementsIf( segments,
                                                          (IndexType) 0,
                                                          segments.getSegmentsCount(),
                                                          std::forward< Condition >( condition ),
                                                          std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forSegments( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   auto segments_view = segments.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = segments_view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Segments, typename Function >
void
forAllSegments( const Segments& segments, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< Segments >::forSegments(
      segments, (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
sequentialForSegments( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   for( IndexType i = begin; i < end; i++ )
      forSegments( segments, i, i + 1, std::forward< Function >( function ) );
}

template< typename Segments, typename Function >
void
sequentialForAllSegments( const Segments& segments, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< Segments >::sequentialForSegments(
      segments, (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

}  // namespace TNL::Algorithms::Segments