// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <gtest/gtest.h>

#include "../TridiagonalMatrixShapes.h"

using TridiagonalMatrixReduceTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::TridiagonalMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::TridiagonalMatrix< float, TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
   TNL::Matrices::TridiagonalMatrix< double, TNL::Devices::Cuda, int >,
   TNL::Matrices::TridiagonalMatrix< float, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TNL::Matrices::TridiagonalMatrix< double, TNL::Devices::Hip, int >,
   TNL::Matrices::TridiagonalMatrix< float, TNL::Devices::Hip, long >
#endif
   >;

template< typename MatrixType >
void
setupTestMatrix( MatrixType& matrix )
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;

   auto view = matrix.getView();

   view.setElement( (IndexType) 0, (IndexType) 0, (RealType) 1.0 );
   view.setElement( (IndexType) 0, (IndexType) 1, (RealType) 2.0 );

   view.setElement( (IndexType) 1, (IndexType) 0, (RealType) 3.0 );
   view.setElement( (IndexType) 1, (IndexType) 1, (RealType) 4.0 );
   view.setElement( (IndexType) 1, (IndexType) 2, (RealType) 5.0 );

   view.setElement( (IndexType) 2, (IndexType) 1, (RealType) 6.0 );
   view.setElement( (IndexType) 2, (IndexType) 2, (RealType) 7.0 );
   view.setElement( (IndexType) 2, (IndexType) 3, (RealType) 8.0 );

   view.setElement( (IndexType) 3, (IndexType) 2, (RealType) 9.0 );
   view.setElement( (IndexType) 3, (IndexType) 3, (RealType) 10.0 );
   view.setElement( (IndexType) 3, (IndexType) 4, (RealType) 11.0 );

   view.setElement( (IndexType) 4, (IndexType) 3, (RealType) 12.0 );
   view.setElement( (IndexType) 4, (IndexType) 4, (RealType) 13.0 );
}

template< typename MatrixType >
void
test_reduceRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      // For TridiagonalMatrix, the second fetch argument is the actual column index. A wrong column
      // index spoils the result, which is checked on the host.
      return columnIdx >= row - 1 && columnIdx <= row + 1 ? value : static_cast< RealType >( 1000 );
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceRows( matrix, (IndexType) 1, (IndexType) 4, fetch, TNL::Plus{}, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   const auto constMatrix( matrix );
   rowSums = 0;

   TNL::Matrices::reduceRows( constMatrix, (IndexType) 1, (IndexType) 4, fetch, TNL::Plus{}, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename MatrixType >
void
test_reduceAllRows_explicit_identity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   // Pass the identity explicitly to exercise the overload with the identity parameter,
   // even though TNL::Plus could deduce it (see test_reduceAllRows_deduced_identity).
   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   const auto constMatrix( matrix );
   rowSums = 0;

   TNL::Matrices::reduceAllRows( constMatrix, fetch, TNL::Plus{}, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );
}

template< typename MatrixType >
void
test_reduceAllRows_deduced_identity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType rowMax( 5, 0 );
   auto rowMaxView = rowMax.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      rowMaxView[ row ] = value;
   };

   // Deduced identity requires a reduction object that exposes getIdentity(),
   // hence we use TNL::Max here instead of a plain lambda.
   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Max{}, keep );

   EXPECT_EQ( rowMax.getElement( 0 ), 2 );
   EXPECT_EQ( rowMax.getElement( 1 ), 5 );
   EXPECT_EQ( rowMax.getElement( 2 ), 8 );
   EXPECT_EQ( rowMax.getElement( 3 ), 11 );
   EXPECT_EQ( rowMax.getElement( 4 ), 13 );

   const auto constMatrix( matrix );
   rowMax = 0;

   TNL::Matrices::reduceAllRows( constMatrix, fetch, TNL::Max{}, keep );

   EXPECT_EQ( rowMax.getElement( 0 ), 2 );
   EXPECT_EQ( rowMax.getElement( 1 ), 5 );
   EXPECT_EQ( rowMax.getElement( 2 ), 8 );
   EXPECT_EQ( rowMax.getElement( 3 ), 11 );
   EXPECT_EQ( rowMax.getElement( 4 ), 13 );
}

template< typename MatrixType >
void
test_reduceRowsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto condition = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };
   auto store = [ = ] __cuda_callable__( IndexType rank, IndexType rowIdx, const RealType& value ) mutable
   {
      rowSumsView[ rowIdx ] = value;
   };

   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceAllRowsIf( matrix, condition, fetch, TNL::Plus{}, store, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );  // 1+2
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );  // 6+7+8
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );  // 12+13

   const auto constMatrix( matrix );
   rowSums = 0;
   TNL::Matrices::reduceRowsIf( constMatrix, (IndexType) 1, (IndexType) 5, condition, fetch, TNL::Plus{}, store, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped by range
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // skipped by condition
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );  // 6+7+8
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped by condition
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );  // 12+13
}

template< typename MatrixType >
void
test_reduceRowsWithArgument_range()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType maxValues( 5, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( 5, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto store = [ = ] __cuda_callable__(
                   IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool emptyRow ) mutable
   {
      maxValuesView[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumnsView[ rowIdx ] = columnIdx;
   };

   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 2 );
   EXPECT_EQ( maxColumns.getElement( 0 ), 1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 11 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

   const auto constMatrix( matrix );
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument(
      constMatrix, (IndexType) 1, (IndexType) 4, fetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 11 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 4 ), -1 );
}

template< typename MatrixType >
void
test_reduceRowsWithArgument_array()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType maxValues( 5, 0 );
   IndexVectorType maxColumns( 5, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto store = [ = ] __cuda_callable__(
                   IndexType idx,
                   IndexType rowIdx,
                   IndexType localIdx,
                   IndexType columnIdx,
                   const RealType& value,
                   bool emptyRow ) mutable
   {
      maxValuesView[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumnsView[ rowIdx ] = columnIdx;
   };

   IndexVectorType rowIndexes{ 1, 2, 4 };
   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceRowsWithArgument( matrix, rowIndexes, fetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 1 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

   const auto constMatrix( matrix );
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument( constMatrix, rowIndexes, fetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 1 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
}

template< typename MatrixType >
void
test_reduceRowsWithArgumentIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   VectorType maxValues( 5, 0 );
   IndexVectorType maxColumns( 5, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto condition = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   auto store = [ = ] __cuda_callable__(
                   IndexType rank,
                   IndexType rowIdx,
                   IndexType localIdx,
                   IndexType columnIdx,
                   const RealType& value,
                   bool emptyRow ) mutable
   {
      maxValuesView[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumnsView[ rowIdx ] = columnIdx;
   };

   // Range variant
   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 11 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

   // Array variant
   const auto constMatrix( matrix );
   IndexVectorType rowIndexes{ 0, 2, 3, 4 };
   auto conditionArray = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgumentIf(
      constMatrix,
      rowIndexes,
      (IndexType) 0,
      rowIndexes.getSize(),
      conditionArray,
      fetch,
      TNL::MaxWithArg{},
      store,
      (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition (rowIdx=0)
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 11 );  // processed
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );  // processed
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
}

// Rectangular (rows > columns) matrix: the last row has only the sub-diagonal element
// (no diagonal, no super-diagonal), which exercises a branch that a square test matrix
// never reaches.
template< typename MatrixType >
void
setupRectangularTestMatrix( MatrixType& matrix )
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;

   auto view = matrix.getView();

   view.setElement( (IndexType) 0, (IndexType) 0, (RealType) 1.0 );
   view.setElement( (IndexType) 0, (IndexType) 1, (RealType) 2.0 );

   view.setElement( (IndexType) 1, (IndexType) 0, (RealType) 3.0 );
   view.setElement( (IndexType) 1, (IndexType) 1, (RealType) 4.0 );
   view.setElement( (IndexType) 1, (IndexType) 2, (RealType) 5.0 );

   view.setElement( (IndexType) 2, (IndexType) 1, (RealType) 6.0 );
   view.setElement( (IndexType) 2, (IndexType) 2, (RealType) 7.0 );
   view.setElement( (IndexType) 2, (IndexType) 3, (RealType) 8.0 );

   view.setElement( (IndexType) 3, (IndexType) 2, (RealType) 9.0 );
   view.setElement( (IndexType) 3, (IndexType) 3, (RealType) 10.0 );

   // Row 4: rowIdx (4) >= columns (4), so only the sub-diagonal element (column 3) exists.
   view.setElement( (IndexType) 4, (IndexType) 3, (RealType) 11.0 );
}

template< typename MatrixType >
void
test_reduceRows_rectangularTailRow()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 4 );
   setupRectangularTestMatrix( matrix );
   auto view = matrix.getView();

   // fetch returns the column index (not the value): if a wrong columnIdx is ever passed
   // to fetch for the sub-diagonal element, the row sum below will not match.
   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& ) -> RealType
   {
      return (RealType) columnIdx;
   };

   VectorType columnIndexSums( 5, 0 );
   auto sumsView = columnIndexSums.getView();
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      sumsView[ row ] = value;
   };

   // Pass the identity explicitly to exercise the overload with the identity parameter.
   TNL::Matrices::reduceAllRows( view, fetch, TNL::Plus{}, keep, (RealType) 0 );

   EXPECT_EQ( columnIndexSums.getElement( 0 ), 1 );  // columns 0 + 1
   EXPECT_EQ( columnIndexSums.getElement( 1 ), 3 );  // columns 0 + 1 + 2
   EXPECT_EQ( columnIndexSums.getElement( 2 ), 6 );  // columns 1 + 2 + 3
   EXPECT_EQ( columnIndexSums.getElement( 3 ), 5 );  // columns 2 + 3
   EXPECT_EQ( columnIndexSums.getElement( 4 ), 3 );  // column 3 only (sub-diagonal tail row)

   // reduceRowsWithArgument: the tail row has a single element, so the stored column index
   // comes straight from that element's fetch() call -- a wrong columnIdx here is not masked
   // by any reduction logic.
   VectorType maxValues( 5, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( 5, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto valueFetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto store = [ = ] __cuda_callable__(
                   IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool emptyRow ) mutable
   {
      maxValuesView[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumnsView[ rowIdx ] = columnIdx;
   };

   TNL::Matrices::reduceAllRowsWithArgument( view, valueFetch, TNL::MaxWithArg{}, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 4 ), 11 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 3 );  // must be the sub-diagonal column, not rowIdx (4)
}

template< typename MatrixType >
void
test_reduceRows_Shapes()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using RealVector = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostRealVector = TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType >;
   using HostIndexVector = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   for( const auto& shape : getTridiagonalTestShapes() ) {
      const IndexType rows = shape.first;
      const IndexType columns = shape.second;
      SCOPED_TRACE( "matrix " + std::to_string( rows ) + "x" + std::to_string( columns ) );
      MatrixType matrix( rows, columns );
      setupTridiagonalTestMatrix( matrix );
      const auto constView = matrix.getConstView();

      // Expected results: the sum of `1000 * value + columnIdx` over the row checks both the values
      // and the column indexes, the maximum is the element with the largest column index.
      std::vector< RealType > expectedSums( rows, 0 );
      std::vector< IndexType > expectedMaxColumns( rows, -1 );
      for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
         for( IndexType localIdx = 0; localIdx < 3; localIdx++ ) {
            const int columnIdx = getTridiagonalColumnIndex( rowIdx, localIdx, columns );
            if( columnIdx >= 0 ) {
               expectedSums[ rowIdx ] += 1000 * getTridiagonalTestValue( rowIdx, columnIdx ) + columnIdx;
               expectedMaxColumns[ rowIdx ] = columnIdx;
            }
         }

      // Row indexes in reverse order
      std::vector< IndexType > hostRowIndexes( rows );
      for( IndexType i = 0; i < rows; i++ )
         hostRowIndexes[ i ] = rows - 1 - i;
      IndexVector rowIndexes( hostRowIndexes );

      RealVector sums( rows );
      IndexVector maxColumns( rows );
      IndexVector maxLocalIdxs( rows );
      RealVector maxValues( rows );
      IndexVector emptyRows( rows );
      auto sums_view = sums.getView();
      auto maxColumns_view = maxColumns.getView();
      auto maxLocalIdxs_view = maxLocalIdxs.getView();
      auto maxValues_view = maxValues.getView();
      auto emptyRows_view = emptyRows.getView();

      auto fetch = [] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return 1000 * value + columnIdx;
      };
      auto valueFetch = [] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      };
      auto store = [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         sums_view[ rowIdx ] = value;
      };
      auto storeWithRowIndexes =
         [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const RealType& value ) mutable
      {
         sums_view[ rowIdx ] = value;
      };
      auto storeWithArgument =
         [ = ] __cuda_callable__(
            IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool emptyRow ) mutable
      {
         maxColumns_view[ rowIdx ] = emptyRow ? -1 : columnIdx;
         maxLocalIdxs_view[ rowIdx ] = emptyRow ? -1 : localIdx;
         maxValues_view[ rowIdx ] = emptyRow ? 0 : value;
         emptyRows_view[ rowIdx ] = emptyRow;
      };
      auto storeWithArgumentAndRowIndexes = [ = ] __cuda_callable__(
                                               IndexType indexOfRowIdx,
                                               IndexType rowIdx,
                                               IndexType localIdx,
                                               IndexType columnIdx,
                                               const RealType& value,
                                               bool emptyRow ) mutable
      {
         maxColumns_view[ rowIdx ] = emptyRow ? -1 : columnIdx;
         maxLocalIdxs_view[ rowIdx ] = emptyRow ? -1 : localIdx;
         maxValues_view[ rowIdx ] = emptyRow ? 0 : value;
         emptyRows_view[ rowIdx ] = emptyRow;
      };
      auto resetMaxima = [ & ]()
      {
         maxColumns.setValue( -2 );
         maxLocalIdxs.setValue( -2 );
         maxValues.setValue( -1 );
         emptyRows.setValue( -1 );
      };

      auto checkSums = [ & ]( const std::vector< bool >& processedRows )
      {
         HostRealVector hostSums;
         hostSums = sums;
         for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
            EXPECT_EQ( hostSums[ rowIdx ], processedRows[ rowIdx ] ? expectedSums[ rowIdx ] : -1 ) << "row " << rowIdx;
      };
      auto checkMaxima = [ & ]()
      {
         HostIndexVector hostColumns;
         HostIndexVector hostLocalIdxs;
         HostRealVector hostValues;
         HostIndexVector hostEmptyRows;
         hostColumns = maxColumns;
         hostLocalIdxs = maxLocalIdxs;
         hostValues = maxValues;
         hostEmptyRows = emptyRows;
         for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ ) {
            const IndexType columnIdx = expectedMaxColumns[ rowIdx ];
            EXPECT_EQ( hostEmptyRows[ rowIdx ], columnIdx < 0 ? 1 : 0 ) << "row " << rowIdx;
            EXPECT_EQ( hostColumns[ rowIdx ], columnIdx ) << "row " << rowIdx;
            EXPECT_EQ( hostLocalIdxs[ rowIdx ], columnIdx < 0 ? -1 : columnIdx - rowIdx + 1 ) << "row " << rowIdx;
            EXPECT_EQ( hostValues[ rowIdx ], columnIdx < 0 ? 0 : getTridiagonalTestValue( rowIdx, columnIdx ) ) << "row " << rowIdx;
         }
      };
      const std::vector< bool > allRows( rows, true );
      std::vector< bool > evenRows( rows );
      for( IndexType i = 0; i < rows; i++ )
         evenRows[ i ] = i % 2 == 0;

      sums.setValue( -1 );
      TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, store );
      checkSums( allRows );

      sums.setValue( -1 );
      TNL::Matrices::reduceAllRows( constView, fetch, TNL::Plus{}, store );
      checkSums( allRows );

      sums.setValue( -1 );
      TNL::Matrices::reduceRows( matrix, rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes );
      checkSums( allRows );

      sums.setValue( -1 );
      TNL::Matrices::reduceAllRowsIf(
         constView,
         [] __cuda_callable__( IndexType rowIdx ) -> bool
         {
            return rowIdx % 2 == 0;
         },
         fetch,
         TNL::Plus{},
         storeWithRowIndexes );
      checkSums( evenRows );

      resetMaxima();
      TNL::Matrices::reduceAllRowsWithArgument( matrix, valueFetch, TNL::MaxWithArg{}, storeWithArgument );
      checkMaxima();

      resetMaxima();
      TNL::Matrices::reduceRowsWithArgument(
         constView, rowIndexes, valueFetch, TNL::MaxWithArg{}, storeWithArgumentAndRowIndexes );
      checkMaxima();
   }
}

// Test fixture
template< typename MatrixType >
class TridiagonalMatrixReduceTest : public ::testing::Test
{
protected:
   using MatrixType_ = MatrixType;
};

TYPED_TEST_SUITE_P( TridiagonalMatrixReduceTest );

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRows )
{
   test_reduceRows< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceAllRows_explicit_identity )
{
   test_reduceAllRows_explicit_identity< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceAllRows_deduced_identity )
{
   test_reduceAllRows_deduced_identity< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRowsIf )
{
   test_reduceRowsIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRowsWithArgument_range )
{
   test_reduceRowsWithArgument_range< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRowsWithArgument_array )
{
   test_reduceRowsWithArgument_array< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRowsWithArgumentIf )
{
   test_reduceRowsWithArgumentIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRows_rectangularTailRow )
{
   test_reduceRows_rectangularTailRow< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixReduceTest, reduceRows_Shapes )
{
   test_reduceRows_Shapes< TypeParam >();
}

REGISTER_TYPED_TEST_SUITE_P(
   TridiagonalMatrixReduceTest,
   reduceRows,
   reduceAllRows_explicit_identity,
   reduceAllRows_deduced_identity,
   reduceRowsIf,
   reduceRowsWithArgument_range,
   reduceRowsWithArgument_array,
   reduceRowsWithArgumentIf,
   reduceRows_rectangularTailRow,
   reduceRows_Shapes );

INSTANTIATE_TYPED_TEST_SUITE_P( TridiagonalMatrix, TridiagonalMatrixReduceTest, TridiagonalMatrixReduceTypes );

#include "../../main.h"
