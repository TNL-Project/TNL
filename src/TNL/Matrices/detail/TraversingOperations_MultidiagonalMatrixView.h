// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../MultidiagonalMatrixView.h"
#include "TraversingOperations.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct TraversingOperations< MultidiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = MultidiagonalMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;
   using RowView = typename MatrixView::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forElements( begin, end, function );
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
      matrix.forElements( begin, end, function );
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
      matrix.forElements( rowIndexes, begin, end, function );
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
      matrix.forElements( rowIndexes, begin, end, function );
   }

   template< typename Array, typename Function >
   static void
   forElements(
      MatrixView& matrix,
      const Array& rowIndexes,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forElements( rowIndexes, function );
   }

   template< typename Array, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forElements( rowIndexes, function );
   }

   template< typename Function >
   static void
   forAllElements( MatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElements( function );
   }

   template< typename Function >
   static void
   forAllElements( const ConstMatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElements( function );
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
      matrix.forElementsIf( begin, end, condition, function );
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
      matrix.forElementsIf( begin, end, condition, function );
   }

   template< typename Condition, typename Function >
   static void
   forAllElementsIf(
      MatrixView& matrix,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElementsIf( condition, function );
   }

   template< typename Condition, typename Function >
   static void
   forAllElementsIf(
      const ConstMatrixView& matrix,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElementsIf( condition, function );
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
      matrix.forRows( begin, end, function );
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
      matrix.forRows( begin, end, function );
   }

   template< typename Function >
   static void
   forAllRows( MatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllRows( function );
   }

   template< typename Function >
   static void
   forAllRows( const ConstMatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllRows( function );
   }

   // MultidiagonalMatrixBase does not provide forRows(rowIndexes, ...) or forRowsIf(...). The
   // following overloads emulate them via Algorithms::parallelFor + getRow, which still avoids
   // re-implementing the multidiagonal indexer logic (getRow encapsulates that).

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
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         auto rowIdx = rowIndexes_view[ idx ];
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         auto rowIdx = rowIndexes_view[ idx ];
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename Array, typename Function >
   static void
   forRows(
      MatrixView& matrix,
      const Array& rowIndexes,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      forRows( matrix, rowIndexes, (IndexType) 0, rowIndexes.getSize(), function, launchConfig );
   }

   template< typename Array, typename Function >
   static void
   forRows(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      forRows( matrix, rowIndexes, (IndexType) 0, rowIndexes.getSize(), function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function >
   static void
   forRowsIf(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      RowCondition&& rowCondition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( rowCondition( rowIdx ) ) {
            auto rowView = matrix.getRow( rowIdx );
            function( rowView );
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function >
   static void
   forRowsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      RowCondition&& rowCondition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( rowCondition( rowIdx ) ) {
            auto rowView = matrix.getRow( rowIdx );
            function( rowView );
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename RowCondition, typename Function >
   static void
   forAllRowsIf(
      MatrixView& matrix,
      RowCondition&& rowCondition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      forRowsIf( matrix, (IndexType) 0, matrix.getRows(), rowCondition, function, launchConfig );
   }

   template< typename RowCondition, typename Function >
   static void
   forAllRowsIf(
      const ConstMatrixView& matrix,
      RowCondition&& rowCondition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      forRowsIf( matrix, (IndexType) 0, matrix.getRows(), rowCondition, function, launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
