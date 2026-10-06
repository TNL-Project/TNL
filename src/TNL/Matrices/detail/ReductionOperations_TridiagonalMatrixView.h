// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../TridiagonalMatrixView.h"
#include "ReductionOperations.h"
#include "ReductionOperationsBase.h"
#include "TridiagonalRowTraversal.h"

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
         forTridiagonalRowElements(
            indexer,
            rowIdx,
            [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx )
            {
               result = reduction( result, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
            } );
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
         forTridiagonalRowElements(
            indexer,
            rowIdx,
            [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx )
            {
               result = reduction( result, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
            } );
         store( rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

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
         const IndexType rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         forTridiagonalRowElements(
            indexer,
            rowIdx,
            [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx )
            {
               result = reduction( result, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
            } );
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
         const IndexType rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         forTridiagonalRowElements(
            indexer,
            rowIdx,
            [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx )
            {
               result = reduction( result, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
            } );
         store( idx, rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }

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

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
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

         forTridiagonalRowElements( indexer, rowIdx, process );
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

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
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

         forTridiagonalRowElements( indexer, rowIdx, process );
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

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
         const IndexType rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
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

         forTridiagonalRowElements( indexer, rowIdx, process );
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
         const IndexType rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;

         auto process = [ & ]( IndexType localIdx, IndexType columnIdx, IndexType globalIdx ) mutable
         {
            auto fetchValue = fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
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

         forTridiagonalRowElements( indexer, rowIdx, process );
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }
};

}  // namespace TNL::Matrices::detail
