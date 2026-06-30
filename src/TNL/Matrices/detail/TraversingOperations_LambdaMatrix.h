// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../LambdaMatrix.h"
#include "TraversingOperations.h"

namespace TNL::Matrices::detail {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
struct TraversingOperations< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
{
   using Matrix = LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >;
   using ConstMatrixView = typename Matrix::ConstViewType;
   using ValueType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using RowView = typename Matrix::RowView;
   using ConstRowView = typename Matrix::ConstRowView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forElements( begin, end, function );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forElements( begin, end, function );
   }

   template< typename Function >
   static void
   forAllElements( Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElements( function );
   }

   template< typename Function >
   static void
   forAllElements( const ConstMatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllElements( function );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forRows( begin, end, function );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forRows( begin, end, function );
   }

   template< typename Function >
   static void
   forAllRows( Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllRows( function );
   }

   template< typename Function >
   static void
   forAllRows( const ConstMatrixView& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      matrix.forAllRows( function );
   }
};

}  // namespace TNL::Matrices::detail
