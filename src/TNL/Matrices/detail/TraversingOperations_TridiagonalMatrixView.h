// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../TridiagonalMatrixView.h"
#include "TraversingOperations.h"
#include "TraversingOperationsBase.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct TraversingOperations< TridiagonalMatrixView< Real, Device, Index, Organization > >
: public TraversingOperationsBase< TridiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = TridiagonalMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;
   using RowView = typename MatrixView::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   // TODO: `launchConfig` is accepted below but never forwarded to Algorithms::parallelFor (see
   // TraversingOperationsBase.h for why). Should eventually be fixed, pending a benchmark.

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
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( rowIdx == 0 ) {
            function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
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
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( rowIdx == 0 ) {
            function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
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
      auto values_view = matrix.getValues().getView();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         if( rowIdx == 0 ) {
            function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
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
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         if( rowIdx == 0 ) {
            function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
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
};

}  // namespace TNL::Matrices::detail
