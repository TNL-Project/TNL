// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../SparseMatrixView.h"
#include "TraversingOperations.h"
#include "TraversingOperationsBase.h"

namespace TNL::Matrices::detail {

template<
   typename Real,
   typename Device,
   typename Index,
   typename MatrixType_,
   template< typename Device_, typename Index_ > class SegmentsView,
   typename ComputeReal >
struct TraversingOperations< SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal > >
: public TraversingOperationsBase< SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal > >
{
   using MatrixView = SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >;
   using ConstMatrixView = SparseMatrixView< const Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;
   using RowView = typename MatrixView::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   // Bring in the array-overloads of forElementsIf from the base class: the range-overloads
   // declared below in this struct would otherwise hide ALL base-class forElementsIf overloads
   // (C++ name hiding), including the array-based ones.
   //
   // The range-overloads declared below use Algorithms::Segments::forElementsIf: a single fused
   // kernel where `condition` is evaluated per row inline during the same traversal that calls
   // `function`. The inherited base-class range-overloads (TraversingOperationsBase) instead
   // build a 0/1 row mask via forAllElements, compact it into a row-index array via
   // Algorithms::compressFast (which involves a blocking device->host sync), and then delegate
   // to the array-overload -- effectively three kernel launches plus a sync instead of one fused
   // kernel. It has not been benchmarked which approach actually performs better in practice
   // (e.g. the mask+compress path launches a second kernel sized to the number of matching rows,
   // which could win when `condition` is rarely true); until that is measured, do not assume the
   // Segments fast path below is strictly superior.
   using TraversingOperationsBase< MatrixView >::forElementsIf;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto columns_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();

      Algorithms::Segments::forElements(
         matrix.getSegments(),
         begin,
         end,
         [ columns_view, values_view, columns, function ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto columns_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();
      Algorithms::Segments::forElements(
         matrix.getSegments(),
         begin,
         end,
         [ columns_view, values_view, columns, function ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      MatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto columns_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();

      Algorithms::Segments::forElements(
         matrix.getSegments(),
         rowIndexes.getConstView( begin, end ),
         [ columns_view, values_view, columns, function ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto columns_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();

      Algorithms::Segments::forElements(
         matrix.getSegments(),
         rowIndexes.getConstView( begin, end ),
         [ function, values_view, columns_view, columns ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto columns_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();
      auto columns = matrix.getColumns();

      Algorithms::Segments::forElementsIf(
         matrix.getSegments(),
         begin,
         end,
         condition,
         [ function, values_view, columns_view, columns ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto columns_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();
      auto columns = matrix.getColumns();
      Algorithms::Segments::forElementsIf(
         matrix.getSegments(),
         begin,
         end,
         condition,
         [ function, values_view, columns_view, columns ] __cuda_callable__(
            IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
         {
            if( localIdx < columns ) {
               if constexpr( MatrixView::isBinary() )
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], static_cast< ValueType >( 1.0 ) );
               else
                  function( segmentIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
            }
         },
         launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto columns_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = RowView( segmentView, values_view, columns_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto columns_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = ConstRowView( segmentView, values_view, columns_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      MatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto columns_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = RowView( segmentView, values_view, columns_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), rowIndexes.getConstView( begin, end ), f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto columns_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();
      using SegmentViewType = typename MatrixView::SegmentsViewType::SegmentViewType;
      auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
      {
         auto rowView = ConstRowView( segmentView, values_view, columns_view );
         function( rowView );
      };
      Algorithms::Segments::forSegments( matrix.getSegments(), rowIndexes.getConstView( begin, end ), f, launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
