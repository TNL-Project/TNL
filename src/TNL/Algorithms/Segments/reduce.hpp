// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "reduce.h"
#include "detail/ReducingOperations.h"

namespace TNL::Algorithms::Segments {

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          typename T >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegments( segments.getConstView(),
                                                                                   begin,
                                                                                   end,
                                                                                   std::forward< Fetch >( fetch ),
                                                                                   std::forward< Reduction >( reduction ),
                                                                                   std::forward< ResultKeeper >( keeper ),
                                                                                   identity,
                                                                                   launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegments( segments.getConstView(),
                                                                                   begin,
                                                                                   end,
                                                                                   std::forward< Fetch >( fetch ),
                                                                                   std::forward< Reduction >( reduction ),
                                                                                   std::forward< ResultKeeper >( keeper ),
                                                                                   Reduction::template getIdentity< Value >(),
                                                                                   launchConfig );
}

template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   reduceSegments( segments,
                   (IndexType) 0,
                   segments.getSegmentsCount(),
                   std::forward< Fetch >( fetch ),
                   std::forward< Reduction >( reduction ),
                   std::forward< ResultKeeper >( keeper ),
                   identity,
                   launchConfig );
}

template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegments( segments,
                   (IndexType) 0,
                   segments.getSegmentsCount(),
                   std::forward< Fetch >( fetch ),
                   std::forward< Reduction >( reduction ),
                   std::forward< ResultKeeper >( keeper ),
                   Reduction::template getIdentity< Value >(),
                   launchConfig );
}
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          typename T >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithSegmentIndexes(
      segments.getConstView(),
      segmentIndexes,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultKeeper >( keeper ),
      identity,
      launchConfig );
}

template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegments( segments.getConstView(),
                                                                                   segmentIndexes,
                                                                                   begin,
                                                                                   end,
                                                                                   std::forward< Fetch >( fetch ),
                                                                                   std::forward< Reduction >( reduction ),
                                                                                   std::forward< ResultKeeper >( keeper ),
                                                                                   Reduction::template getIdentity< Value >(),
                                                                                   launchConfig );
}

template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          typename T >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig )
{
   reduceSegments( segments,
                   segmentIndexes,
                   (typename Segments::IndexType) 0,
                   segmentIndexes.getSize(),
                   std::forward< Fetch >( fetch ),
                   std::forward< Reduction >( reduction ),
                   std::forward< ResultKeeper >( keeper ),
                   identity,
                   launchConfig );
}

template< typename Segments, typename Array, typename Fetch, typename Reduction, typename ResultKeeper, typename T >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegments( segments,
                   segmentIndexes,
                   (typename Segments::IndexType) 0,
                   segmentIndexes.getSize(),
                   std::forward< Fetch >( fetch ),
                   std::forward< Reduction >( reduction ),
                   std::forward< ResultKeeper >( keeper ),
                   Reduction::template getIdentity< Value >(),
                   launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            const Value& identity,
                            LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithArgument(
      segments.getConstView(),
      begin,
      end,
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultKeeper >( keeper ),
      identity,
      launchConfig );
}

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithArgument(
      segments.getConstView(),
      begin,
      end,
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultKeeper >( keeper ),
      Reduction::template getIdentity< Value >(),
      launchConfig );
}

template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   reduceSegmentsWithArgument( segments,
                               (IndexType) 0,
                               segments.getSegmentsCount(),
                               std::forward< Fetch >( fetch ),
                               std::forward< Reduction >( reduction ),
                               std::forward< ResultKeeper >( keeper ),
                               identity,
                               launchConfig );
}

template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegmentsWithArgument( segments,
                               (IndexType) 0,
                               segments.getSegmentsCount(),
                               std::forward< Fetch >( fetch ),
                               std::forward< Reduction >( reduction ),
                               std::forward< ResultKeeper >( keeper ),
                               Reduction::template getIdentity< Value >(),
                               launchConfig );
}

}  // namespace TNL::Algorithms::Segments
