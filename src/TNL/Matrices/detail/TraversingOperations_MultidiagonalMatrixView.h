// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../MultidiagonalMatrixView.h"
#include "TraversingOperations.h"
#include "TraversingOperationsBase.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct TraversingOperations< MultidiagonalMatrixView< Real, Device, Index, Organization > >
: public TraversingOperationsBase< MultidiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = MultidiagonalMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;
   using RowView = typename MatrixView::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   // TODO: `launchConfig` is accepted below only for consistency with the other matrix types and it is
   // not used. Most of it describes how threads are mapped to segments, but the rows of this matrix type
   // are not stored in segments and each row is processed by one thread of Algorithms::parallelFor.
   // Only its block size could be forwarded to parallelFor on GPUs, which should be benchmarked first.

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
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
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
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
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
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
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
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         auto rowIdx = rowIndexes_view[ idx ];
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }
};

}  // namespace TNL::Matrices::detail
