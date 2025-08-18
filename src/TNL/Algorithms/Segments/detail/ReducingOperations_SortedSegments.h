// SPDX-FileComment: This file is part of TNL - Template Numerical Library
// (https://tnl-project.org/) SPDX-License-Identifier: MIT

#pragma once

#include "FetchLambdaAdapter.h"
#include "ReducingOperationsBase.h"
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

template< typename EmbeddedSegmentsView_ >
struct ReducingOperations< SortedSegmentsView< EmbeddedSegmentsView_ > >
: public ReducingOperationsBase< SortedSegmentsView< EmbeddedSegmentsView_ > >
{
   using EmbeddedSegmentsView = EmbeddedSegmentsView_;
   using DeviceType = typename EmbeddedSegmentsView::DeviceType;
   using IndexType = typename EmbeddedSegmentsView::IndexType;
   using ViewType = SortedSegmentsView< EmbeddedSegmentsView_ >;
   using ConstViewType = typename ViewType::ConstViewType;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO Fetch&& does not work with nvcc
                   Reduction reduction,  // TODO Reduction&& does not work with nvcc
                   ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;

      if( end <= begin )
         return;

      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      if( begin == 0 && end == segments.getSegmentsCount() ) {
         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperations< EmbeddedSegmentsView >::reduceSegments(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
               {
                  TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
                  TNL_ASSERT_LT(
                     segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
                  TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
                  TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                                 inverseSegmentsPermutationView.getSize(),
                                 "Transformed segment index is larger than number of segments." );

                  return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
               },
               reduction,
               [ = ] __cuda_callable__( IndexType segmentIdx, const ReturnType& result ) mutable
               {
                  TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
                  TNL_ASSERT_LT(
                     segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
                  TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
                  TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                                 inverseSegmentsPermutationView.getSize(),
                                 "Transformed segment index is larger than number of segments." );

                  keeper( inverseSegmentsPermutationView[ segmentIdx ], result );
               },
               identity,
               launchConfig );
         }
         else {  // argumentCount< Fetch >() == 1
            ReducingOperations< EmbeddedSegmentsView >::reduceSegments(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               fetch,
               reduction,
               [ = ] __cuda_callable__( IndexType segmentIdx, const ReturnType& result ) mutable
               {
                  TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
                  TNL_ASSERT_LT(
                     segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
                  TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
                  TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                                 inverseSegmentsPermutationView.getSize(),
                                 "Transformed segment index is larger than number of segments." );

                  keeper( inverseSegmentsPermutationView[ segmentIdx ], result );
               },
               identity,
               launchConfig );
         }
      }
      else {
         if constexpr( argumentCount< Fetch >() == 3 ) {
            auto fetch_ = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
               TNL_ASSERT_LT(
                  segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
               TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
               TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                              inverseSegmentsPermutationView.getSize(),
                              "Transformed segment index is larger than number of segments." );

               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            };
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               0,
               end - begin,
               fetch_,
               reduction,
               [ = ] __cuda_callable__( IndexType segmentIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
               {
                  TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
                  TNL_ASSERT_LT(
                     segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
                  TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
                  TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                                 inverseSegmentsPermutationView.getSize(),
                                 "Transformed segment index is larger than number of segments." );

                  keeper( inverseSegmentsPermutationView[ segmentIdx ], result );
               },
               identity,
               launchConfig );
         }
         else {  // argumentCount< Fetch >() == 1
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               0,
               end - begin,
               fetch,
               reduction,
               [ = ] __cuda_callable__( IndexType segmentdIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
               {
                  TNL_ASSERT_GE( segmentIdx, 0, "Segment index is negative." );
                  TNL_ASSERT_LT(
                     segmentIdx, inverseSegmentsPermutationView.getSize(), "Segment index is larger than number of segments." );
                  TNL_ASSERT_GE( inverseSegmentsPermutationView[ segmentIdx ], 0, "Transformed segment index is negative." );
                  TNL_ASSERT_LT( inverseSegmentsPermutationView[ segmentIdx ],
                                 inverseSegmentsPermutationView.getSize(),
                                 "Transformed segment index is larger than number of segments." );

                  keeper( inverseSegmentsPermutationView[ segmentIdx ], result );
               },
               identity,
               launchConfig );
         }
      }
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     IndexBegin begin,
                                     IndexEnd end,
                                     Fetch fetch,          // TODO Fetch&& does not work with nvcc
                                     Reduction reduction,  // TODO Reduction&& does not work with nvcc
                                     ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;

      if( end <= begin )
         return;

      Array aux( end - begin );
      auto segmentIndexesView = segmentIndexes.getConstView();
      auto segmentsPermutationView = segments.getSegmentsPermutationView();
      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      aux.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            TNL_ASSERT_LT( i + begin, segmentIndexesView.getSize(), "" );
            value = segmentsPermutationView[ segmentIndexesView[ i + begin ] ];
         } );

      if constexpr( argumentCount< Fetch >() == 3 ) {
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            0,
            aux.getSize(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            },
            reduction,
            [ = ] __cuda_callable__( IndexType segmentIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], result );
            },
            identity,
            launchConfig );
      }
      else {  // argumentCount< Fetch >() == 1
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            0,
            aux.getSize(),
            fetch,
            reduction,
            [ = ] __cuda_callable__( IndexType segmentIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], result );
            },
            identity,
            launchConfig );
      }
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                               Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                               ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;

      if( end <= begin )
         return;

      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      if( begin == 0 && end == segments.getSegmentsCount() ) {
         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithArgument(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
               {
                  return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
               },
               reduction,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
               {
                  keeper( inverseSegmentsPermutationView[ segmentIdx ], argument, result );
               },
               identity,
               launchConfig );
         }
         else {  // argumentCount< Fetch >() == 1
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithArgument(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               fetch,
               reduction,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
               {
                  keeper( inverseSegmentsPermutationView[ segmentIdx ], argument, result );
               },
               identity,
               launchConfig );
         }
      }
      else {
         if constexpr( argumentCount< Fetch >() == 3 ) {
            auto fetch_ = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            };
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               0,
               end - begin,
               fetch_,
               reduction,
               [ = ] __cuda_callable__(
                  IndexType segmentIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
               {
                  keeper( inverseSegmentsPermutationView[ segmentIdx ], argument, result );
               },
               identity,
               launchConfig );
         }
         else {  // argumentCount< Fetch >() == 1
            ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               0,
               end - begin,
               fetch,
               reduction,
               [ = ] __cuda_callable__(
                  IndexType segmentdIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
               {
                  keeper( inverseSegmentsPermutationView[ segmentIdx ], argument, result );
               },
               identity,
               launchConfig );
         }
      }
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      IndexBegin begin,
      IndexEnd end,
      Fetch fetch,          // TODO: Fetch&& fetch does not work here with CUDA
      Reduction reduction,  // TODO: Reduction&& reduction does not work here with CUDA
      ResultKeeper keeper,  // TODO: ResultKeeper&& keeper does not work here with CUDA
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;

      if( end <= begin )
         return;

      Array aux( end - begin );
      auto segmentIndexesView = segmentIndexes.getConstView();
      auto segmentsPermutationView = segments.getSegmentsPermutationView();
      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      aux.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            TNL_ASSERT_LT( i + begin, segmentIndexesView.getSize(), "" );
            value = segmentsPermutationView[ segmentIndexesView[ i + begin ] ];
         } );

      if constexpr( argumentCount< Fetch >() == 3 ) {
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            0,
            aux.getSize(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            },
            reduction,
            [ = ] __cuda_callable__(
               IndexType segmentIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], argument, result );
            },
            identity,
            launchConfig );
      }
      else {  // argumentCount< Fetch >() == 1
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            0,
            aux.getSize(),
            fetch,
            reduction,
            [ = ] __cuda_callable__(
               IndexType segmentIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], argument, result );
            },
            identity,
            launchConfig );
      }
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsIf( const ConstViewType& segments,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultKeeper&& keeper,
                     const Value& identity,
                     LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      if( end <= begin )
         return;

      VectorType conditions( segments.getSegmentsCount() );
      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      auto conditionsView = conditions.getView();
      Algorithms::parallelFor< DeviceType >( begin,
                                             end,
                                             [ = ] __cuda_callable__( IndexType i ) mutable
                                             {
                                                conditionsView[ i ] = condition( inverseSegmentsPermutationView[ i + begin ] );
                                             } );

      auto indexes = compressFast< VectorType >( conditions );
      if constexpr( argumentCount< Fetch >() == 3 ) {
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
            segments.getEmbeddedSegmentsView(),
            indexes.getConstView(),
            0,
            indexes.getSize(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            },
            reduction,
            [ = ] __cuda_callable__( IndexType segmentIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], result );
            },
            identity,
            launchConfig );
      }
      else {  // argumentCount< Fetch >() == 1
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexes(
            segments.getEmbeddedSegmentsView(),
            indexes.getConstView(),
            0,
            indexes.getSize(),
            fetch,
            reduction,
            [ = ] __cuda_callable__( IndexType segmentIdx_idx, IndexType segmentIdx, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], result );
            },
            identity,
            launchConfig );
      }
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsIfWithArgument( const ConstViewType& segments,
                                 IndexBegin begin,
                                 IndexEnd end,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultKeeper&& keeper,
                                 const Value& identity,
                                 LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      if( end <= begin )
         return;

      VectorType conditions( segments.getSegmentsCount() );
      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      auto conditionsView = conditions.getView();
      Algorithms::parallelFor< DeviceType >( begin,
                                             end,
                                             [ = ] __cuda_callable__( IndexType i ) mutable
                                             {
                                                conditionsView[ i ] = condition( inverseSegmentsPermutationView[ i + begin ] );
                                             } );

      auto indexes = compressFast< VectorType >( conditions );
      if constexpr( argumentCount< Fetch >() == 3 ) {
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
            segments.getEmbeddedSegmentsView(),
            indexes.getConstView(),
            0,
            indexes.getSize(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               return fetch( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            },
            reduction,
            [ = ] __cuda_callable__(
               IndexType segmentIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], argument, result );
            },
            identity,
            launchConfig );
      }
      else {  // argumentCount< Fetch >() == 1
         ReducingOperations< EmbeddedSegmentsView >::reduceSegmentsWithSegmentIndexesAndArgument(
            segments.getEmbeddedSegmentsView(),
            indexes.getConstView(),
            0,
            indexes.getSize(),
            fetch,
            reduction,
            [ = ] __cuda_callable__(
               IndexType segmentIdx_idx, IndexType segmentIdx, IndexType argument, const ReturnType& result ) mutable
            {
               keeper( segmentIdx_idx, inverseSegmentsPermutationView[ segmentIdx ], argument, result );
            },
            identity,
            launchConfig );
      }
   }
};

}  // namespace TNL::Algorithms::Segments::detail
