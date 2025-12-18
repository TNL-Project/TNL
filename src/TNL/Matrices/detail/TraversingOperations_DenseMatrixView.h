// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../DenseMatrixView.h"
#include "TraversingOperations.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct TraversingOperations< DenseMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = DenseMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;
   using RowView = typename MatrixView::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( MatrixView& matrix,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns )
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
      };
      Algorithms::Segments::forElements( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstMatrixView& matrix,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns )
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
      };
      Algorithms::Segments::forElements( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( MatrixView& matrix,
                const Array& rowIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns )
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
      };
      Algorithms::Segments::forElements( matrix.getSegments(), rowIndexes, begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstMatrixView& matrix,
                const Array& rowIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns )
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
      };
      Algorithms::Segments::forElements( matrix.getSegments(), rowIndexes, begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( MatrixView& matrix,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns ) {
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
         }
      };
      Algorithms::Segments::forElementsIf( matrix.getSegments(), begin, end, condition, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstMatrixView& matrix,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();
      auto f = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < columns )
            function( segmentIdx, localIdx, localIdx, values_view[ globalIdx ] );
      };
      Algorithms::Segments::forElementsIf( matrix.getSegments(), begin, end, condition, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows( MatrixView& matrix,
            IndexBegin begin,
            IndexEnd end,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = RowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows( const ConstMatrixView& matrix,
            IndexBegin begin,
            IndexEnd end,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = ConstRowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows( MatrixView& matrix,
            const Array& rowIndexes,
            IndexBegin begin,
            IndexEnd end,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = RowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), rowIndexes, begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows( const ConstMatrixView& matrix,
            const Array& rowIndexes,
            IndexBegin begin,
            IndexEnd end,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         const auto rowView = ConstRowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), rowIndexes, begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function >
   static void
   forRowsIf( MatrixView& matrix,
              IndexBegin begin,
              IndexEnd end,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = RowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegmentsIf( matrix.getSegments(), begin, end, rowCondition, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function >
   static void
   forRowsIf( const ConstMatrixView& matrix,
              IndexBegin begin,
              IndexEnd end,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( const SegmentViewType& segmentView ) mutable
      {
         auto rowView = ConstRowView( segmentView, values_view );
         function( rowView );
      };
      Algorithms::Segments::forSegmentsIf( matrix.getSegments(), begin, end, rowCondition, f, launchConfig );
   }
};

}  //namespace TNL::Matrices::detail
