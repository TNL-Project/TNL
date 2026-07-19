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

   // TODO: `launchConfig` is accepted below but never forwarded to Algorithms::parallelFor (see
   // TraversingOperationsBase.h for why). Should eventually be fixed, pending a benchmark.

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      Matrix& matrix,
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
               function( rowIdx, localIdx, columnIdx, elementValue );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
               function( rowIdx, localIdx, columnIdx, elementValue );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements(
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue != 0.0 )
               function( rowIdx, localIdx, columnIdx, elementValue );
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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue != 0.0 )
               function( rowIdx, localIdx, columnIdx, elementValue );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
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
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         auto rowIdx = rowIndexes_view[ idx ];
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
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         auto rowIdx = rowIndexes_view[ idx ];
         auto rowView = matrix.getRow( rowIdx );
         function( rowView );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }
};

}  // namespace TNL::Matrices::detail
