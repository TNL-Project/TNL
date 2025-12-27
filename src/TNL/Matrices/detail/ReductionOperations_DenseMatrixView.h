// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keep, identity, launchConfig );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      Algorithms::Segments::reduceSegments(
         matrix.getSegments(), rowIndexes, fetchWrapper, reduction, keep, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep >
   static void
   reduceRows( MatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceRows(
         matrix, rowIndexes, std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Keep >
   static void
   reduceRows( const ConstMatrixView& matrix,
               const Array& rowIndexes,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceRows(
         matrix, rowIndexes, std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), identity, launchConfig );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( rowIdx, localIdx, localIdx, value );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();
      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( rowIdx, localIdx, localIdx, value );
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
      auto values_view = matrix.getValues().getView();
      const auto columns = matrix.getColumns();

      auto fetchWrapper =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, localIdx, localIdx, value );
      };

      Algorithms::Segments::reduceSegmentsWithArgument( matrix.getSegments(),
                                                        rowIndexes,
                                                        0,
                                                        rowIndexes.getSize(),
                                                        fetchWrapper,
                                                        reduction,
                                                        keepWrapper,
                                                        identity,
                                                        launchConfig );
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
      const auto values_view = matrix.getValues().getConstView();
      const auto columns = matrix.getColumns();

      auto fetchWrapper = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue
      {
         if( localIdx < columns )
            return fetch( rowIdx, localIdx, values_view[ globalIdx ] );
         return identity;
      };
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, localIdx, localIdx, value );
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
   reduceRowsIfWithArgument( MatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, localIdx, localIdx, value );
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
   reduceRowsIfWithArgument( const ConstMatrixView& matrix,
                             IndexBegin begin,
                             IndexEnd end,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
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
      auto keepWrapper = [ = ] __cuda_callable__(
                            IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, const FetchValue& value ) mutable
      {
         keep( indexOfRowIdx, rowIdx, localIdx, localIdx, value );
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
};
}  // namespace TNL::Matrices::detail
