// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename EmbeddedSegmentsView_ >
struct TraversingOperations< SortedSegmentsView< EmbeddedSegmentsView_ > >
: public TraversingOperationsBase< SortedSegmentsView< EmbeddedSegmentsView_ > >
{
   using EmbeddedSegmentsView = EmbeddedSegmentsView_;
   using DeviceType = typename EmbeddedSegmentsView::DeviceType;
   using IndexType = typename EmbeddedSegmentsView::IndexType;
   using ViewType = SortedSegmentsView< EmbeddedSegmentsView >;
   using ConstViewType = typename ViewType::ConstViewType;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;

      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      if( begin == 0 && end == segments.getSegmentCount() ) {
         if constexpr( argumentCount< Function >() == 3 ) {
            TraversingOperations< EmbeddedSegmentsView >::forElements(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
               },
               launchConfig );
         }
         else {  // argumentCount< Function >() == 2
            TraversingOperations< EmbeddedSegmentsView >::forElements(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], globalIdx );
               },
               launchConfig );
         }
      }
      else {
         if constexpr( argumentCount< Function >() == 3 ) {
            TraversingOperations< EmbeddedSegmentsView >::forElements(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
               },
               launchConfig );
         }
         else {  // argumentCount< Function >() == 2
            TraversingOperations< EmbeddedSegmentsView >::forElements(
               segments.getEmbeddedSegmentsView(),
               segments.getSegmentsPermutationView().getConstView( begin, end ),
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], globalIdx );
               },
               launchConfig );
         }
      }
   }

   template< typename Array, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      if( segmentIndexes.getSize() == 0 )
         return;

      Containers::Array< IndexType, DeviceType, IndexType > aux( segmentIndexes.getSize() );
      auto segmentIndexesView = segmentIndexes.getConstView();
      auto segmentsPermutationView = segments.getSegmentsPermutationView();
      auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
      aux.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            TNL_ASSERT_LT( i, segmentIndexesView.getSize(), "" );
            value = segmentsPermutationView[ segmentIndexesView[ i ] ];
         } );

      if constexpr( argumentCount< Function >() == 3 ) {
         TraversingOperations< EmbeddedSegmentsView >::forElements(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
            {
               function( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
            },
            launchConfig );
      }
      else {  // argumentCount< Function >() == 2
         TraversingOperations< EmbeddedSegmentsView >::forElements(
            segments.getEmbeddedSegmentsView(),
            aux.getConstView(),
            [ = ] __cuda_callable__( IndexType segmentIdx, IndexType globalIdx ) mutable
            {
               function( inverseSegmentsPermutationView[ segmentIdx ], globalIdx );
            },
            launchConfig );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Function&& function,
                  LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      if( begin == 0 && end == segments.getSegmentCount() ) {
         auto inverseSegmentsPermutationView = segments.getInverseSegmentsPermutationView();
         if constexpr( argumentCount< Function >() == 3 ) {
            TraversingOperations< EmbeddedSegmentsView >::forElementsIf(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
               {
                  return condition( inverseSegmentsPermutationView[ segmentIdx ] );
               },
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], localIdx, globalIdx );
               },
               launchConfig );
         }
         else {  // argumentCount< Function >() == 2
            TraversingOperations< EmbeddedSegmentsView >::forElementsIf(
               segments.getEmbeddedSegmentsView(),
               begin,
               end,
               [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
               {
                  return condition( inverseSegmentsPermutationView[ segmentIdx ] );
               },
               [ = ] __cuda_callable__( IndexType segmentIdx, IndexType globalIdx ) mutable
               {
                  function( inverseSegmentsPermutationView[ segmentIdx ], globalIdx );
               },
               launchConfig );
         }
      }
      else
         forElementsIfSparse( segments, begin, end, condition, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSparse( const ConstViewType& segments,
                        IndexBegin begin,
                        IndexEnd end,
                        Condition&& condition,
                        Function&& function,
                        LaunchConfiguration launchConfig )
   {
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      if( end <= begin )
         return;

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i );
         } );

      auto indexes = compressFast< VectorType >( conditions );
      indexes += begin;
      forElements( segments, indexes, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forSegments( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      using SegmentView = typename ConstViewType::SegmentViewType;

      if( end <= begin )
         return;

      auto segments_view = segments.getConstView();
      if( begin == 0 && end == segments.getSegmentCount() ) {
         TraversingOperations< EmbeddedSegmentsView >::forSegments(
            segments.getEmbeddedSegmentsView(),
            0,
            segments.getSegmentCount(),
            [ = ] __cuda_callable__( SegmentView & segment ) mutable
            {
               segment.setSegmentIndex( segments_view.getInverseSegmentsPermutationView()[ segment.getSegmentIndex() ] );
               function( segment );
            },
            launchConfig );
      }
      else {
         Containers::Vector< IndexType, DeviceType, IndexType > segmentIndexes( end - begin );
         auto segmentsPermutationView = segments.getConstView().getSegmentsPermutationView();
         segmentIndexes.forAllElements(
            [ = ] __cuda_callable__( IndexType i, IndexType & value )
            {
               TNL_ASSERT_LT( i + begin, segments.getSegmentCount(), "" );
               value = segmentsPermutationView[ i + begin ];
            } );
         TraversingOperations< EmbeddedSegmentsView >::forSegments(
            segments.getEmbeddedSegmentsView(),
            segmentIndexes,
            [ = ] __cuda_callable__( SegmentView & segment ) mutable
            {
               segment.setSegmentIndex( segments_view.getInverseSegmentsPermutationView()[ segment.getSegmentIndex() ] );
               function( segment );
            },
            launchConfig );
      }
   }

   template< typename Array, typename Function >
   static void
   forSegments( const ConstViewType& segments,
                const Array& segmentIndexes,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      using SegmentView = typename ConstViewType::SegmentViewType;

      if( segmentIndexes.getSize() == 0 )
         return;

      auto segments_view = segments.getConstView();
      Containers::Vector< IndexType, DeviceType, IndexType > transformedSegmentIndexes( segmentIndexes.getSize() );
      auto segmentsPermutationView = segments.getConstView().getSegmentsPermutationView();
      auto segmentIndexesView = segmentIndexes.getConstView();
      transformedSegmentIndexes.forElements( 0,
                                             segmentIndexes.getSize(),
                                             [ = ] __cuda_callable__( IndexType i, IndexType & value )
                                             {
                                                value = segmentsPermutationView[ segmentIndexesView[ i ] ];
                                             } );

      TraversingOperations< EmbeddedSegmentsView >::forSegments(
         segments.getEmbeddedSegmentsView(),
         transformedSegmentIndexes,
         [ = ] __cuda_callable__( SegmentView & segment ) mutable
         {
            segment.setSegmentIndex( segments_view.getInverseSegmentsPermutationView()[ segment.getSegmentIndex() ] );
            function( segment );
         },
         launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename SegmentCondition, typename Function >
   static void
   forSegmentsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  SegmentCondition&& segmentCondition,
                  Function&& function,
                  LaunchConfiguration launchConfig )
   {
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      if( end <= begin )
         return;

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = segmentCondition( i );
         } );

      auto indexes = compressFast< VectorType >( conditions );
      indexes += begin;
      forSegments( segments, indexes, function, launchConfig );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
