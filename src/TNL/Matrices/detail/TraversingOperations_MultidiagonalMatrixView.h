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

   // ===================== forElements (range) =====================

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== forElements (array) =====================

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
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== forElementsIf =====================

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
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== forRows (range) =====================

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== forRows (array) =====================

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
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== forRowsIf =====================

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
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }
};

}  // namespace TNL::Matrices::detail
