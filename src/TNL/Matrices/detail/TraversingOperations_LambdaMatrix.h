// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../LambdaMatrix.h"
#include "TraversingOperations.h"
#include "TraversingOperationsBase.h"

namespace TNL::Matrices::detail {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
struct TraversingOperations< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
: public TraversingOperationsBase< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
{
   using Matrix = LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >;
   using ConstMatrixView = typename Matrix::ConstViewType;
   using ValueType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using RowView = typename Matrix::RowView;
   using ConstRowView = typename ConstMatrixView::ConstRowView;

   // TODO: `launchConfig` is accepted below only for consistency with the other matrix types and it is
   // not used. Most of it describes how threads are mapped to segments, but the rows of this matrix type
   // are not stored in segments and each row is processed by one thread of Algorithms::parallelFor.
   // Only its block size could be forwarded to parallelFor on GPUs, which should be benchmarked first.

   // A lambda matrix cannot be modified, so only the const overloads are provided. They accept also
   // non-const matrices and pass the matrix elements to the user function as constant values.

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue != 0.0 )
               function( rowIdx, localIdx, columnIdx, static_cast< const Real& >( elementValue ) );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const IndexType rowIdx = rowIndexes_view[ idx ];
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue != 0.0 )
               function( rowIdx, localIdx, columnIdx, static_cast< const Real& >( elementValue ) );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forRows(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      TNL_ASSERT_GE( begin, 0, "Parameter 'begin' must be non-negative." );
      TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );
      TNL_ASSERT_LE(
         end, rowIndexes.getSize(), "Parameter 'end' must be lower or equal to the size of the array of row indexes." );
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const IndexType rowIdx = rowIndexes_view[ idx ];
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }
};

}  // namespace TNL::Matrices::detail
