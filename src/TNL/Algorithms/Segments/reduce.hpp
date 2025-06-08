// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/reduce.h>
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
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithSegmentIndexes(
      segments.getConstView(),
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
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          typename T >
static void
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultKeeper&& keeper,
                  const Value& identity,
                  LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsIf( segments.getConstView(),
                                                                                     begin,
                                                                                     end,
                                                                                     std::forward< Condition >( condition ),
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     std::forward< Reduction >( reduction ),
                                                                                     std::forward< ResultKeeper >( keeper ),
                                                                                     Reduction::template getIdentity< Value >(),
                                                                                     launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T >
static void
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsIf( segments.getConstView(),
                                                                                     begin,
                                                                                     end,
                                                                                     std::forward< Condition >( condition ),
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     std::forward< Reduction >( reduction ),
                                                                                     std::forward< ResultKeeper >( keeper ),
                                                                                     Reduction::template getIdentity< Value >(),
                                                                                     launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
static void
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultKeeper&& keeper,
                     const Value& identity,
                     LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   reduceSegmentsIf( segments,
                     (IndexType) 0,
                     segments.getSegmentsCount(),
                     std::forward< Condition >( condition ),
                     std::forward< Fetch >( fetch ),
                     std::forward< Reduction >( reduction ),
                     std::forward< ResultKeeper >( keeper ),
                     identity,
                     launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultKeeper&& keeper,
                     LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegmentsIf( segments,
                     (IndexType) 0,
                     segments.getSegmentsCount(),
                     std::forward< Condition >( condition ),
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
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            const Value& identity,
                            LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithSegmentIndexesAndArgument(
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
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsWithSegmentIndexesAndArgument(
      segments.getConstView(),
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
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            const Value& identity,
                            LaunchConfiguration launchConfig )
{
   reduceSegmentsWithArgument( segments,
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
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegmentsWithArgument( segments,
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
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          typename T >
static void
reduceSegmentsIfWithArgument( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultKeeper&& keeper,
                              const Value& identity,
                              LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsIfWithArgument(
      segments.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultKeeper >( keeper ),
      Reduction::template getIdentity< Value >(),
      launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T >
static void
reduceSegmentsIfWithArgument( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultKeeper&& keeper,
                              LaunchConfiguration launchConfig )
{
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegmentsIfWithArgument(
      segments.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultKeeper >( keeper ),
      Reduction::template getIdentity< Value >(),
      launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
static void
reduceAllSegmentsIfWithArgument( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultKeeper&& keeper,
                                 const Value& identity,
                                 LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   reduceSegmentsIfWithArgument( segments,
                                 (IndexType) 0,
                                 segments.getSegmentsCount(),
                                 std::forward< Condition >( condition ),
                                 std::forward< Fetch >( fetch ),
                                 std::forward< Reduction >( reduction ),
                                 std::forward< ResultKeeper >( keeper ),
                                 identity,
                                 launchConfig );
}

template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsIfWithArgument( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultKeeper&& keeper,
                                 LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType;
   reduceSegmentsIfWithArgument( segments,
                                 (IndexType) 0,
                                 segments.getSegmentsCount(),
                                 std::forward< Condition >( condition ),
                                 std::forward< Fetch >( fetch ),
                                 std::forward< Reduction >( reduction ),
                                 std::forward< ResultKeeper >( keeper ),
                                 Reduction::template getIdentity< Value >(),
                                 launchConfig );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename SegmentsReductionValue,
          typename FinalReductionValue,
          typename T >
static FinalReductionValue
reduce( const Segments& segments,
        IndexBegin begin,
        IndexEnd end,
        SegmentFetch&& segmentFetch,
        SegmentReduction&& segmentReduction,
        FinalFetch&& finalFetch,
        FinalReduction&& finalReduction,
        const SegmentsReductionValue& segmentsReductionIdentity,
        const FinalReductionValue& finalReductionIdentity,
        LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;

   if( end <= begin )
      return finalReductionIdentity;

   // Allocate array for segment results
   Containers::Vector< SegmentsReductionValue, DeviceType, IndexType > segmentResults( end - begin );
   auto segmentResultsView = segmentResults.getView();

   // First reduce within segments
   reduceSegments(
      segments,
      begin,
      end,
      std::forward< SegmentFetch >( segmentFetch ),
      std::forward< SegmentReduction >( segmentReduction ),
      [ begin, segmentResultsView ] __cuda_callable__( IndexType segmentIdx, const SegmentsReductionValue& value ) mutable
      {
         segmentResultsView[ segmentIdx - begin ] = value;
      },
      segmentsReductionIdentity,
      launchConfig );

   // Then reduce segment results using the result fetch and reduction
   return TNL::Algorithms::reduce< DeviceType >( (IndexType) 0,
                                                 segmentResults.getSize(),
                                                 [ segmentResultsView, &finalFetch ] __cuda_callable__( IndexType idx ) mutable
                                                 {
                                                    return finalFetch( segmentResultsView[ idx ] );
                                                 },
                                                 std::forward< FinalReduction >( finalReduction ),
                                                 finalReductionIdentity );
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename T >
static typename detail::FetchLambdaAdapter< typename Segments::IndexType, SegmentFetch >::ReturnType
reduce( const Segments& segments,
        IndexBegin begin,
        IndexEnd end,
        SegmentFetch&& segmentFetch,
        SegmentReduction&& segmentReduction,
        FinalFetch&& finalFetch,
        FinalReduction&& finalReduction,
        LaunchConfiguration launchConfig )
{
   using SegmentValue = typename detail::FetchLambdaAdapter< typename Segments::IndexType, SegmentFetch >::ReturnType;
   using FinalValue = typename detail::FetchLambdaAdapter< typename Segments::IndexType, FinalFetch >::ReturnType;
   return reduce( segments,
                  begin,
                  end,
                  std::forward< SegmentFetch >( segmentFetch ),
                  std::forward< SegmentReduction >( segmentReduction ),
                  std::forward< FinalFetch >( finalFetch ),
                  std::forward< FinalReduction >( finalReduction ),
                  SegmentReduction::template getIdentity< SegmentValue >(),
                  FinalReduction::template getIdentity< FinalValue >(),
                  launchConfig );
}

template< typename Segments,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename SegmentsReductionValue,
          typename FinalReductionValue >
static FinalReductionValue
reduceAll( const Segments& segments,
           SegmentFetch&& segmentFetch,
           SegmentReduction&& segmentReduction,
           FinalFetch&& finalFetch,
           FinalReduction&& finalReduction,
           const SegmentsReductionValue& segmentsReductionIdentity,
           const FinalReductionValue& finalReductionIdentity,
           LaunchConfiguration launchConfig )
{
   return reduce( segments,
                  (typename Segments::IndexType) 0,
                  segments.getSegmentsCount(),
                  std::forward< SegmentFetch >( segmentFetch ),
                  std::forward< SegmentReduction >( segmentReduction ),
                  std::forward< FinalFetch >( finalFetch ),
                  std::forward< FinalReduction >( finalReduction ),
                  segmentsReductionIdentity,
                  finalReductionIdentity,
                  launchConfig );
}

template< typename Segments, typename SegmentFetch, typename SegmentReduction, typename FinalFetch, typename FinalReduction >
static typename detail::FetchLambdaAdapter< typename Segments::IndexType, FinalFetch >::ReturnType
reduceAll( const Segments& segments,
           SegmentFetch&& segmentFetch,
           SegmentReduction&& segmentReduction,
           FinalFetch&& finalFetch,
           FinalReduction&& finalReduction,
           LaunchConfiguration launchConfig )
{
   using SegmentValue = typename detail::FetchLambdaAdapter< typename Segments::IndexType, SegmentFetch >::ReturnType;
   using FinalValue = typename detail::FetchLambdaAdapter< typename Segments::IndexType, FinalFetch >::ReturnType;
   return reduce( segments,
                  (typename Segments::IndexType) 0,
                  segments.getSegmentsCount(),
                  std::forward< SegmentFetch >( segmentFetch ),
                  std::forward< SegmentReduction >( segmentReduction ),
                  std::forward< FinalFetch >( finalFetch ),
                  std::forward< FinalReduction >( finalReduction ),
                  SegmentReduction::template getIdentity< SegmentValue >(),
                  FinalReduction::template getIdentity< FinalValue >(),
                  launchConfig );
}

}  // namespace TNL::Algorithms::Segments
