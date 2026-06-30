// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/TypeTraits.h>
#include "../TridiagonalMatrixView.h"
#include "ReductionOperations.h"

namespace TNL::Matrices::detail {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct ReductionOperations< TridiagonalMatrixView< Real, Device, Index, Organization > >
{
   using MatrixView = TridiagonalMatrixView< Real, Device, Index, Organization >;
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
      matrix.reduceRows( begin, end, fetch, reduction, store, identity );
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
      matrix.reduceRows( begin, end, fetch, reduction, store, identity );
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
      matrix.reduceAllRows( fetch, reduction, store, identity );
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
      matrix.reduceAllRows( fetch, reduction, store, identity );
   }
};

}  // namespace TNL::Matrices::detail
