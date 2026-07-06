// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Array.h>
#include "../MultidiagonalMatrixView.h"
#include "ReductionOperations.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct ReductionOperations< MultidiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = MultidiagonalMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;

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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         store( rowIdx, sum );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         store( rowIdx, sum );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         store( idx, rowIdx, sum );
      };
      Algorithms::parallelFor< DeviceType >( (IndexType) 0, rowIndexes.getSize(), f );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         store( idx, rowIdx, sum );
      };
      Algorithms::parallelFor< DeviceType >( (IndexType) 0, rowIndexes.getSize(), f );
   }

   // ===================== reduceRowsIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, sum );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         FetchValue sum = identity;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns )
               sum = reduction( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, sum );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( (IndexType) 0, rowIndexes.getSize(), f );
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
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( (IndexType) 0, rowIndexes.getSize(), f );
   }

   // ===================== reduceRowsWithArgumentIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      MatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         if( ! condition( rowIdx ) )
            return;
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
   }

   // ===================== reduceRowsWithArgumentIf (array) =====================

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      MatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      auto values_view = matrix.getValues().getView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         if( ! condition( idx ) )
            return;
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
   }

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Containers::Array< IndexType, DeviceType > counterArray( 1 );
      counterArray = 0;
      auto counter = counterArray.getView();
      const auto values_view = matrix.getValues().getConstView();
      const auto diagonalOffsets_view = matrix.getDiagonalOffsets().getConstView();
      const IndexType diagonalsCount = matrix.getDiagonalsCount();
      const IndexType columns = matrix.getColumns();
      const auto indexer = matrix.getIndexer();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         if( ! condition( idx ) )
            return;
         const auto rowIdx = rowIndexes_view[ idx ];
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool empty = true;
         for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
            const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
            if( columnIdx >= 0 && columnIdx < columns ) {
               auto fetchValue = fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
               if( empty ) {
                  result = fetchValue;
                  resultLocalIdx = localIdx;
                  resultColumnIdx = columnIdx;
                  empty = false;
               }
               else {
                  auto prev = resultLocalIdx;
                  reduction( result, fetchValue, resultLocalIdx, localIdx );
                  if( resultLocalIdx != prev )
                     resultColumnIdx = columnIdx;
               }
            }
         }
         const auto rank = Algorithms::AtomicOperations< DeviceType >::add( counter[ 0 ], (IndexType) 1 );
         store( rank, rowIdx, resultLocalIdx, resultColumnIdx, result, empty );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
      return counterArray.getElement( 0 );
   }
};

}  // namespace TNL::Matrices::detail
