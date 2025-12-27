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

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRows( MatrixView& matrix,
               IndexBegin begin,
               IndexEnd end,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
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
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keep, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRows( const ConstMatrixView& matrix,
               IndexBegin begin,
               IndexEnd end,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
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
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keep, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRows( MatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, value );
      };

      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRows( const ConstMatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, value );
      };

      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsIf( MatrixView& matrix,
                 IndexBegin begin,
                 IndexEnd end,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
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
      Algorithms::Segments::reduceSegmentsIf( matrix.getSegments(),
                                              begin,
                                              end,
                                              std::forward< Condition >( condition ),
                                              fetchWrapper,
                                              reduction,
                                              keep,
                                              identity,
                                              launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsIf( const ConstMatrixView& matrix,
                 IndexBegin begin,
                 IndexEnd end,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
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
      Algorithms::Segments::reduceSegmentsIf( matrix.getSegments(),
                                              begin,
                                              end,
                                              std::forward< Condition >( condition ),
                                              fetchWrapper,
                                              reduction,
                                              keep,
                                              identity,
                                              launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRowsWithArgument( MatrixView& matrix,
                           IndexBegin begin,
                           IndexEnd end,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRowsWithArgument( const ConstMatrixView& matrix,
                           IndexBegin begin,
                           IndexEnd end,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRowsWithArgument( MatrixView& matrix,
                           const Array& rowIndexes,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
   static void
   reduceRowsWithArgument( const ConstMatrixView& matrix,
                           const Array& rowIndexes,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsWithArgumentIf( MatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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

      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                          begin,
                                                          end,
                                                          std::forward< Condition >( condition ),
                                                          fetchWrapper,
                                                          reduction,
                                                          keepWrapper,
                                                          identity,
                                                          launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsWithArgumentIf( const ConstMatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                          begin,
                                                          end,
                                                          std::forward< Condition >( condition ),
                                                          fetchWrapper,
                                                          reduction,
                                                          keepWrapper,
                                                          identity,
                                                          launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsWithArgumentIf( MatrixView& matrix,
                             const Array& rowIndexes,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                          rowIndexes.getConstView( begin, end ),
                                                          std::forward< Condition >( condition ),
                                                          fetchWrapper,
                                                          reduction,
                                                          keepWrapper,
                                                          identity,
                                                          launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Keep,
             typename FetchValue >
   static void
   reduceRowsWithArgumentIf( const ConstMatrixView& matrix,
                             const Array& rowIndexes,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         const auto columnIdx = columnIndexes_view[ segmentsView.getGlobalIndex( rowIdx, localIdx ) ];
         keep( indexOfRowIdx, rowIdx, localIdx, columnIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgumentIf( matrix.getSegments(),
                                                          rowIndexes.getConstView( begin, end ),
                                                          std::forward< Condition >( condition ),
                                                          fetchWrapper,
                                                          reduction,
                                                          keepWrapper,
                                                          identity,
                                                          launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
