// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include "detail/TraversingOperations.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function, LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Segments::ConstViewType >::forElements(
      segments.getConstView(), begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Segments, typename Function >
void
forAllElements( const Segments& segments, Function&& function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   forElements( segments, (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Segments::ConstViewType >::forElements(
      segments.getConstView(), segmentIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, Function function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   detail::TraversingOperations< typename Segments::ConstViewType >::forElements( segments.getConstView(),
                                                                                  segmentIndexes,
                                                                                  (IndexType) 0,
                                                                                  segmentIndexes.getSize(),
                                                                                  std::forward< Function >( function ),
                                                                                  launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments& segments,
               IndexBegin begin,
               IndexEnd end,
               Condition condition,
               Function function,
               LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Segments::ConstViewType >::forElementsIf( segments.getConstView(),
                                                                                    begin,
                                                                                    end,
                                                                                    std::forward< Condition >( condition ),
                                                                                    std::forward< Function >( function ),
                                                                                    launchConfig );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments& segments, Condition condition, Function function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   detail::TraversingOperations< typename Segments::ConstViewType >::forElementsIf( segments.getConstView(),
                                                                                    (IndexType) 0,
                                                                                    segments.getSegmentsCount(),
                                                                                    std::forward< Condition >( condition ),
                                                                                    std::forward< Function >( function ),
                                                                                    launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIfSparse( const Segments& segments,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition condition,
                     Function function,
                     LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   detail::TraversingOperations< typename Segments::ConstViewType >::forElementsIfSparse(
      segments.getConstView(),
      (IndexType) 0,
      segments.getSegmentsCount(),
      std::forward< Condition >( condition ),
      std::forward< Function >( function ),
      launchConfig );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIfSparse( const Segments& segments, Condition condition, Function function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   forElementsIfSparse( segments.getConstView(),
                        (IndexType) 0,
                        segments.getSegmentsCount(),
                        std::forward< Condition >( condition ),
                        std::forward< Function >( function ),
                        launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forSegments( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   auto segments_view = segments.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = segments_view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with current
                                                            // implementation of parallelFor
}

template< typename Segments, typename Function >
void
forAllSegments( const Segments& segments, Function&& function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   forSegments(
      segments.getConstView(), (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forSegments( const Segments& segments,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   auto segments_view = segments.getConstView();
   auto segmentIndexes_view = segmentIndexes.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx_idx ) mutable
   {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
      TNL_ASSERT_LT( segmentIndexes_view[ segmentIdx_idx ], segments_view.getSegmentsCount(), "" );
      auto segment = segments_view.getSegmentView( segmentIndexes_view[ segmentIdx_idx ] );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with current
                                                            // implementation of parallelFor
}

template< typename Segments, typename Array, typename Function, typename T >
void
forSegments( const Segments& segments, const Array& segmentIndexes, Function&& function, LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   forSegments(
      segments, segmentIndexes, (IndexType) 0, segmentIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename SegmentCondition, typename Function, typename T >
void
forSegmentsIf( const Segments& segments,
               IndexBegin begin,
               IndexEnd end,
               SegmentCondition&& segmentCondition,
               Function&& function,
               LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   auto segments_view = segments.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      if( segmentCondition( segmentIdx ) )
         function( segments_view.getSegmentView( segmentIdx ) );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with current
                                                            // implementation of parallelFor
}

template< typename Segments, typename SegmentCondition, typename Function >
void
forAllSegmentsIf( const Segments& segments,
                  SegmentCondition&& segmentCondition,
                  Function&& function,
                  LaunchConfiguration launchConfig )
{
   forSegmentsIf( segments.getConstView(),
                  (typename Segments::IndexType) 0,
                  segments.getSegmentsCount(),
                  std::forward< SegmentCondition >( segmentCondition ),
                  std::forward< Function >( function ),
                  launchConfig );
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
   sequentialForSegments( segments.getConstView(), 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

}  // namespace TNL::Algorithms::Segments
