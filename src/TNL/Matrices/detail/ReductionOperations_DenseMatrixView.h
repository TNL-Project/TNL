// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/Vector.h>
#include "../DenseMatrixView.h"
#include "ReductionOperations.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct ReductionOperations< DenseMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = DenseMatrixView< Real, Device, Index, Organization >;
   using ConstMatrixView = typename MatrixView::ConstViewType;
   using ValueType = typename MatrixView::RealType;
   using DeviceType = typename MatrixView::DeviceType;
   using IndexType = typename MatrixView::IndexType;

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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, store, identity, launchConfig );
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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, store, identity, launchConfig );
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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, store, identity, launchConfig );
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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store >
   static void
   reduceRows(
      MatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceRows(
         matrix,
         rowIndexes,
         std::forward< Fetch >( fetch ),
         reduction,
         std::forward< Store >( store ),
         identity,
         launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store >
   static void
   reduceRows(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceRows(
         matrix,
         rowIndexes,
         std::forward< Fetch >( fetch ),
         reduction,
         std::forward< Store >( store ),
         identity,
         launchConfig );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      return Algorithms::Segments::reduceSegmentsIf(
         matrix.getSegments(),
         begin,
         end,
         std::forward< Condition >( condition ),
         fetchWrapper,
         reduction,
         store,
         identity,
         launchConfig );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };

      return Algorithms::Segments::reduceSegmentsIf(
         matrix.getSegments(),
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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( rowIdx, localIdx, localIdx, value, emptySegment );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
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
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( rowIdx, localIdx, localIdx, value, emptySegment );
      };
      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), begin, end, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
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
      const auto columns = matrix.getColumns();

      auto fetchWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
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
      const auto columns = matrix.getColumns();

      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf(
         matrix.getSegments(),
         begin,
         end,
         std::forward< Condition >( condition ),
         fetchWrapper,
         reduction,
         keepWrapper,
         identity,
         launchConfig );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      return Algorithms::Segments::reduceSegmentsWithArgumentIf(
         matrix.getSegments(),
         begin,
         end,
         std::forward< Condition >( condition ),
         fetchWrapper,
         reduction,
         keepWrapper,
         identity,
         launchConfig );
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
      if( end <= begin )
         return 0;

      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Evaluate conditions and compress to get filtered indices
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );

      if( filteredIndices.getSize() == 0 )
         return 0;

      // Gather the filtered row indexes
      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      // Call reduceSegmentsWithArgument with the filtered row indexes
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), filteredRowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );

      return filteredRowIndexes.getSize();
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
      if( end <= begin )
         return 0;

      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Evaluate conditions and compress to get filtered indices
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );

      if( filteredIndices.getSize() == 0 )
         return 0;

      // Gather the filtered row indexes
      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      // Call reduceSegmentsWithArgument with the filtered row indexes
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper =
         [ = ] __cuda_callable__(
            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value, bool emptySegment ) mutable
      {
         store( indexOfRowIdx, rowIdx, localIdx, localIdx, value, emptySegment );
      };

      Algorithms::Segments::reduceSegmentsWithArgument(
         matrix.getSegments(), filteredRowIndexes, fetchWrapper, reduction, keepWrapper, identity, launchConfig );

      return filteredRowIndexes.getSize();
   }
};
}  // namespace TNL::Matrices::detail
