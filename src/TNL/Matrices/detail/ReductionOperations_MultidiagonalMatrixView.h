// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
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
      // MultidiagonalMatrixBase::reduceRows calls reduce( sum, value ) without using the return value,
      // expecting the reduction to modify sum in-place. Wrap the provided reduction so that standard
      // function objects such as TNL::Plus work transparently.
      auto reductionWrapper = [ reduction ] __cuda_callable__( auto& sum, const auto& value )
      {
         sum = reduction( sum, value );
      };
      matrix.reduceRows( begin, end, fetch, reductionWrapper, store, identity );
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
      auto reductionWrapper = [ reduction ] __cuda_callable__( auto& sum, const auto& value )
      {
         sum = reduction( sum, value );
      };
      matrix.reduceRows( begin, end, fetch, reductionWrapper, store, identity );
   }

   template< typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceAllRows(
      MatrixView& matrix,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto reductionWrapper = [ reduction ] __cuda_callable__( auto& sum, const auto& value )
      {
         sum = reduction( sum, value );
      };
      matrix.reduceAllRows( fetch, reductionWrapper, store, identity );
   }

   template< typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceAllRows(
      const ConstMatrixView& matrix,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto reductionWrapper = [ reduction ] __cuda_callable__( auto& sum, const auto& value )
      {
         sum = reduction( sum, value );
      };
      matrix.reduceAllRows( fetch, reductionWrapper, store, identity );
   }
};

}  // namespace TNL::Matrices::detail
