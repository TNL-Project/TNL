// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Matrices/MatrixBase.h>
#include "../SparseMatrixView.h"
#include "ReductionOperations.h"

namespace TNL::Matrices::detail {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename Device_, typename Index_ > class SegmentsView,
          typename ComputeReal >
struct ReductionOperations< SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal > >
{
   using MatrixView = SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >;
   using ConstMatrixView = SparseMatrixView< const Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows( MatrixView& matrix,
               IndexBegin begin,
               IndexEnd end,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], (ValueType) 1 );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, store, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows( const ConstMatrixView& matrix,
               IndexBegin begin,
               IndexEnd end,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      const auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         const auto columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, (ValueType) 1 );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows( MatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getView();
      auto values_view = matrix.getValues().getView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], (ValueType) 1 );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      auto storeWrapper = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
      {
         store( indexOfRowIdx, rowIdx, value );
      };

      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows( const ConstMatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      const auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      const auto values_view = matrix.getValues().getConstView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         const auto columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, (ValueType) 1 );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      auto storeWrapper = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
      {
         store( indexOfRowIdx, rowIdx, value );
      };

      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsIf( MatrixView& matrix,
                 IndexBegin begin,
                 IndexEnd end,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], (ValueType) 1 );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      return Algorithms::Segments::reduceSegmentsIf( matrix.getSegments(),
                                                     begin,
                                                     end,
                                                     std::forward< Condition >( condition ),
                                                     fetchWrapper,
                                                     reduction,
                                                     store,
                                                     identity,
                                                     launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsIf( const ConstMatrixView& matrix,
                 IndexBegin begin,
                 IndexEnd end,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         IndexType columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, (ValueType) 1 );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      return Algorithms::Segments::reduceSegmentsIf( matrix.getSegments(),
                                                     begin,
                                                     end,
                                                     std::forward< Condition >( condition ),
                                                     fetchWrapper,
                                                     reduction,
                                                     store,
                                                     identity,
                                                     launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument( MatrixView& matrix,
                           IndexBegin begin,
                           IndexEnd end,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], (ValueType) 1 );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            TNL_ASSERT_LT( rowIdx, matrix.getRows(), "Row index out of bounds in reduceRowsWithArgument." );
            TNL_ASSERT_LT( localIdx,
                           segmentsView.getSegmentSize( rowIdx ),
                           "Local index out of bounds for segment in reduceRowsWithArgument." );
            TNL_ASSERT_LT( segmentsView.getGlobalIndex( rowIdx, localIdx ),
                           columnIndexes_view.getSize(),
                           "Global index out of bounds for columnIndexes_view in reduceRowsWithArgument." );
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument( const ConstMatrixView& matrix,
                           IndexBegin begin,
                           IndexEnd end,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      const auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         const auto columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, (ValueType) 1 );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument( MatrixView& matrix,
                           const Array& rowIndexes,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getView();
      const auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            return fetch( rowIdx,
                          columnIndexes_view[ globalIdx ],
                          values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                                       // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument( const ConstMatrixView& matrix,
                           const Array& rowIndexes,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         IndexType columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, (ValueType) 1 );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, storeWrapper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf( MatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], ValueType( 1.0 ) );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };

      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                                 begin,
                                                                 end,
                                                                 std::forward< Condition >( condition ),
                                                                 fetchWrapper,
                                                                 reduction,
                                                                 storeWrapper,
                                                                 identity,
                                                                 launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf( const ConstMatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         const auto columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, ValueType( 1.0 ) );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                                 begin,
                                                                 end,
                                                                 std::forward< Condition >( condition ),
                                                                 fetchWrapper,
                                                                 reduction,
                                                                 storeWrapper,
                                                                 identity,
                                                                 launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf( MatrixView& matrix,
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
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( columnIndexes_view[ globalIdx ] != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIndexes_view[ globalIdx ], ValueType( 1.0 ) );
            else
               return fetch(
                  rowIdx,
                  columnIndexes_view[ globalIdx ],
                  values_view[ globalIdx ] );  // For non-constant matrix, we allows modification of columns indexes
                                               // and values during the data fetching for the sake of kernels merging.
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                                 rowIndexes.getConstView( begin, end ),
                                                                 std::forward< Condition >( condition ),
                                                                 fetchWrapper,
                                                                 reduction,
                                                                 storeWrapper,
                                                                 identity,
                                                                 launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf( const ConstMatrixView& matrix,
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
      constexpr IndexType paddingIndex = Matrices::paddingIndex< IndexType >;
      auto columnIndexes_view = matrix.getColumnIndexes().getConstView();
      auto values_view = matrix.getValues().getConstView();
      auto segmentsView = matrix.getSegments();

      auto fetchWrapper = [ fetch, values_view, columnIndexes_view, identity ] __cuda_callable__(
                             IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         const auto columnIdx = columnIndexes_view[ globalIdx ];
         if( columnIdx != paddingIndex ) {
            if constexpr( MatrixView::isBinary() )
               return fetch( rowIdx, columnIdx, ValueType( 1.0 ) );
            else
               return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
         }
         return identity;
      };
      auto storeWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         if( ! emptySegment ) {
            const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
            store( indexOfRowIdx, rowIdx, localIdx, columnIdx, value, emptySegment );
         }
         else {
            store( indexOfRowIdx, rowIdx, localIdx, IndexType( 0 ), value, emptySegment );
         }
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                                 rowIndexes.getConstView( begin, end ),
                                                                 std::forward< Condition >( condition ),
                                                                 fetchWrapper,
                                                                 reduction,
                                                                 storeWrapper,
                                                                 identity,
                                                                 launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
