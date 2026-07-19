// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "../LambdaMatrix.h"
#include "ReductionOperations.h"
#include "ReductionOperationsBase.h"

namespace TNL::Matrices::detail {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
struct ReductionOperations< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
: public ReductionOperationsBase< LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index > >
{
   using Matrix = LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >;
   using ConstMatrixView = typename Matrix::ConstViewType;
   using ValueType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   // TODO: `launchConfig` is accepted below but never forwarded to Algorithms::parallelFor (see
   // ReductionOperationsBase.h for why). Should eventually be fixed, pending a benchmark.

   // ===================== reduceRows (range) =====================

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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            FetchValue fetchValue = identity;
            if( elementValue != 0.0 )
               fetchValue = fetch( rowIdx, columnIdx, elementValue );
            result = reduction( result, fetchValue );
         }
         store( rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            FetchValue fetchValue = identity;
            if( elementValue != 0.0 )
               fetchValue = fetch( rowIdx, columnIdx, elementValue );
            result = reduction( result, fetchValue );
         }
         store( rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== reduceRows (array) =====================

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRows(
      Matrix& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
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
         FetchValue result = identity;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            FetchValue fetchValue = identity;
            if( elementValue != 0.0 )
               fetchValue = fetch( rowIdx, columnIdx, elementValue );
            result = reduction( result, fetchValue );
         }
         store( idx, rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            FetchValue fetchValue = identity;
            if( elementValue != 0.0 )
               fetchValue = fetch( rowIdx, columnIdx, elementValue );
            result = reduction( result, fetchValue );
         }
         store( idx, rowIdx, result );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }

   // ===================== reduceRowsWithArgument (range) =====================

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue == 0.0 )
               continue;
            auto fetchValue = fetch( rowIdx, columnIdx, elementValue );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue == 0.0 )
               continue;
            auto fetchValue = fetch( rowIdx, columnIdx, elementValue );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         }
         store( rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( begin, end, f );
   }

   // ===================== reduceRowsWithArgument (array) =====================

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceRowsWithArgument(
      Matrix& matrix,
      const Array& rowIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
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
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue == 0.0 )
               continue;
            auto fetchValue = fetch( rowIdx, columnIdx, elementValue );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
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
      const IndexType rows = matrix.getRows();
      const IndexType columns = matrix.getColumns();
      auto rowLengths = matrix.getCompressedRowLengthsLambda();
      auto matrixElements = matrix.getMatrixElementsLambda();
      auto rowIndexes_view = rowIndexes.getConstView();
      auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         const auto rowIdx = rowIndexes_view[ idx ];
         const IndexType rowLength = rowLengths( rows, columns, rowIdx );
         FetchValue result = identity;
         IndexType resultLocalIdx = 0;
         IndexType resultColumnIdx = 0;
         bool emptyRow = true;
         for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
            IndexType columnIdx( 0 );
            Real elementValue( 0.0 );
            matrixElements( rows, columns, rowIdx, localIdx, columnIdx, elementValue );
            if( elementValue == 0.0 )
               continue;
            auto fetchValue = fetch( rowIdx, columnIdx, elementValue );
            if( emptyRow ) {
               result = fetchValue;
               resultLocalIdx = localIdx;
               resultColumnIdx = columnIdx;
               emptyRow = false;
            }
            else {
               auto prev = resultLocalIdx;
               reduction( result, fetchValue, resultLocalIdx, localIdx );
               if( resultLocalIdx != prev )
                  resultColumnIdx = columnIdx;
            }
         }
         store( idx, rowIdx, resultLocalIdx, resultColumnIdx, result, emptyRow );
      };
      Algorithms::parallelFor< DeviceType >( 0, rowIndexes.getSize(), f );
   }
};

}  // namespace TNL::Matrices::detail
