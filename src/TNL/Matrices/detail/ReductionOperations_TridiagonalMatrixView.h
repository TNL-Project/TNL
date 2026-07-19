// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../TridiagonalMatrixView.h"
#include "ReductionOperations.h"
#include "ReductionOperationsBase.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct ReductionOperations< TridiagonalMatrixView< Real, Device, Index, Organization > >
: public ReductionOperationsBase< TridiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = TridiagonalMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;

   // TODO: `launchConfig` is accepted below but never forwarded to Algorithms::parallelFor (see
   // ReductionOperationsBase.h for why). Should eventually be fixed, pending a benchmark.

   // ===================== reduceRows (range) =====================

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         if( rowIdx == 0 ) {
            result = reduction( result, fetch( 0, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
            result = reduction( result, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         }
         else if( rowIdx < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         }
         else {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         }
         store( rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         if( rowIdx == 0 ) {
            result = reduction( result, fetch( 0, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
            result = reduction( result, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         }
         else if( rowIdx < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         }
         else {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         }
         store( rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== reduceRows (array) =====================

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      MatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         if( rowIdx == 0 ) {
            result = reduction( result, fetch( 0, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
            result = reduction( result, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         }
         else if( rowIdx < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         }
         else {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         }
         store( idx, rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         if( rowIdx == 0 ) {
            result = reduction( result, fetch( 0, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
            result = reduction( result, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         }
         else if( rowIdx < indexer.getColumns() ) {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
            result = reduction( result, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         }
         else {
            result = reduction( result, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         }
         store( idx, rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }

   // ===================== reduceRowsWithArgument (range) =====================

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, ValueType& val ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, val );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         };

         if( rowIdx == 0 ) {
            process( 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            process( 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            process( 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, const ValueType& val ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, val );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         };

         if( rowIdx == 0 ) {
            process( 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            process( 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            process( 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== reduceRowsWithArgument (array) =====================

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      MatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto values_view = matrix.getValues().getView();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, ValueType& val ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, val );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         };

         if( rowIdx == 0 ) {
            process( 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            process( 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            process( 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto values_view = matrix.getValues().getConstView();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, const ValueType& val ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, val );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         };

         if( rowIdx == 0 ) {
            process( 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
            process( 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
         }
         else if( rowIdx + 1 < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
            process( 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
         }
         else if( rowIdx < indexer.getColumns() ) {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
            process( 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         }
         else {
            process( 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }
};

}  // namespace TNL::Matrices::detail
