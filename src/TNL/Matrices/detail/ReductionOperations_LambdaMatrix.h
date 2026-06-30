// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../LambdaMatrix.h"
#include "ReductionOperations.h"

namespace TNL::Matrices::detail {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
struct ReductionOperations< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
{
   using Matrix = LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >;
   using ConstMatrixView = typename Matrix::ConstViewType;
   using ValueType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      Matrix& matrix,
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
      Matrix& matrix,
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
