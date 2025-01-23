// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include "detail/SegmentsOperations.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments,
             IndexBegin begin,
             IndexEnd end,
             const LaunchConfiguration& launchConfig,
             Function&& function )
{
   detail::SegmentsOperations< typename Segments::ConstViewType >::forElements(
      segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function )
{
   LaunchConfiguration launchConfig;
   forElements( segments, begin, end, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename Function >
void
forAllElements( const Segments& segments, const LaunchConfiguration& launchConfig, Function&& function )
{
   using IndexType = typename Segments::IndexType;
   forElements( segments, (IndexType) 0, segments.getSegmentsCount(), launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename Function >
void
forAllElements( const Segments& segments, Function&& function )
{
   LaunchConfiguration launchConfig;
   forAllElements( segments, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             const LaunchConfiguration& launchConfig,
             Function&& function )
{
   detail::SegmentsOperations< typename Segments::ConstViewType >::forElements(
      segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function function )
{
   LaunchConfiguration launchConfig;
   forElements( segments, segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, const LaunchConfiguration& launchConfig, Function function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< typename Segments::ConstViewType >::forElements( segments.getConstView(),
                                                                                segmentIndexes,
                                                                                (IndexType) 0,
                                                                                segmentIndexes.getSize(),
                                                                                launchConfig,
                                                                                std::forward< Function >( function ) );
}

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments& segments, const Array& segmentIndexes, Function function )
{
   LaunchConfiguration launchConfig;
   forElements( segments, segmentIndexes, launchConfig, std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments& segments,
               IndexBegin begin,
               IndexEnd end,
               const LaunchConfiguration& launchConfig,
               Condition condition,
               Function function )
{
   detail::SegmentsOperations< typename Segments::ConstViewType >::forElementsIf( segments.getConstView(),
                                                                                  begin,
                                                                                  end,
                                                                                  launchConfig,
                                                                                  std::forward< Condition >( condition ),
                                                                                  std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
{
   LaunchConfiguration launchConfig;
   forElementsIf(
      segments, begin, end, launchConfig, std::forward< Condition >( condition ), std::forward< Function >( function ) );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments& segments, const LaunchConfiguration& launchConfig, Condition condition, Function function )
{
   using IndexType = typename Segments::IndexType;
   detail::SegmentsOperations< typename Segments::ConstViewType >::forElementsIf( segments.getConstView(),
                                                                                  (IndexType) 0,
                                                                                  segments.getSegmentsCount(),
                                                                                  launchConfig,
                                                                                  std::forward< Condition >( condition ),
                                                                                  std::forward< Function >( function ) );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments& segments, Condition condition, Function function )
{
   LaunchConfiguration launchConfig;
   forAllElementsIf( segments, launchConfig, std::forward< Condition >( condition ), std::forward< Function >( function ) );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIfSparse( const Segments& segments,
                     IndexBegin begin,
                     IndexEnd end,
                     const LaunchConfiguration& launchConfig,
                     Condition condition,
                     Function function )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

   VectorType conditions( end - begin );
   conditions.forAllElements(
      [ = ] __cuda_callable__( IndexType i, IndexType & value )
      {
         value = condition( i + begin );
      } );

   auto indexes = compressFast< VectorType >( conditions );
   forElements( segments, indexes, launchConfig, function );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIfSparse( const Segments& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
{
   LaunchConfiguration launchConfig;
   forElementsIfSparse(
      segments, begin, end, launchConfig, std::forward< Condition >( condition ), std::forward< Function >( function ) );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIfSparse( const Segments& segments,
                        const LaunchConfiguration& launchConfig,
                        Condition condition,
                        Function function )
{
   using IndexType = typename Segments::IndexType;
   forElementsIfSparse( segments.getConstView(),
                        (IndexType) 0,
                        segments.getSegmentsCount(),
                        launchConfig,
                        std::forward< Condition >( condition ),
                        std::forward< Function >( function ) );
}

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIfSparse( const Segments& segments, Condition condition, Function function )
{
   LaunchConfiguration launchConfig;
   forAllElementsIfSparse(
      segments, launchConfig, std::forward< Condition >( condition ), std::forward< Function >( function ) );
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
   detail::SegmentsOperations< typename Segments::ConstViewType >::forSegments(
      segments.getConstView(), (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
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
   detail::SegmentsOperations< typename Segments::ConstViewType >::sequentialForSegments(
      segments.getConstView(), (IndexType) 0, segments.getSegmentsCount(), std::forward< Function >( function ) );
}

}  // namespace TNL::Algorithms::Segments
