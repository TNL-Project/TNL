// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct TraversingOperationsBaseline
{
   using ViewType = typename Segments::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSparse( const ConstViewType& segments,
                        IndexBegin begin,
                        IndexEnd end,
                        Condition condition,
                        Function function,
                        LaunchConfiguration launchConfig )
   {
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i + begin );
         } );

      auto indexes = compressFast< VectorType >( conditions );
      indexes += begin;
      forElements( segments, indexes, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      auto segments_view = segments.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
      {
         auto segment = segments_view.getSegmentView( segmentIdx );
         function( segment );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with
                                                               // current implementation of parallelFor
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
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
      Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with
                                                               // current implementation of parallelFor
   }

   template< typename IndexBegin, typename IndexEnd, typename SegmentCondition, typename Function >
   static void
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
      Algorithms::parallelFor< DeviceType >( begin, end, f );  // TODO: Add launchConfig - it seems it does not work with
                                                               // current implementation of parallelFor
   }
};

}  //namespace TNL::Algorithms::Segments::detail
