// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/compress.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct TraversingOperations;

template< typename Segments >
struct TraversingOperationsBase
{
   using ViewType = typename Segments::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   /**
    * This method does the same as forElementsIf, but with a materialization of the condition results.
    * It seems to be slower than forElementsIf, so it may serve mainly as a fallback for
    * for segments where forElementsIf is not implemented.
    */
   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forSelectedElements( const ConstViewType& segments,
                        IndexBegin begin,
                        IndexEnd end,
                        Condition&& condition,
                        Function&& function,
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
      // TODO: Add launchConfig - it seems it does not work with current implementation of parallelFor
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename Array, typename Function >
   static void
   forSegments( const Segments& segments, const Array& segmentIndexes, Function&& function, LaunchConfiguration launchConfig )
   {
      using IndexType = typename Segments::IndexType;
      using DeviceType = typename Segments::DeviceType;
      auto segments_view = segments.getConstView();
      auto segmentIndexes_view = segmentIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx_idx ) mutable
      {
         TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
         TNL_ASSERT_LT( segmentIndexes_view[ segmentIdx_idx ], segments_view.getSegmentCount(), "" );
         auto segment = segments_view.getSegmentView( segmentIndexes_view[ segmentIdx_idx ] );
         function( segment );
      };
      // TODO: Add launchConfig - it seems it does not work with current implementation of parallelFor
      Algorithms::parallelFor< DeviceType >( 0, segmentIndexes.getSize(), f );
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
         auto segment = segments_view.getSegmentView( segmentIdx );
         if( segmentCondition( segmentIdx ) )
            function( segment );
      };
      // TODO: Add launchConfig - it seems it does not work with current implementation of parallelFor
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }
};

}  // namespace TNL::Algorithms::Segments::detail
