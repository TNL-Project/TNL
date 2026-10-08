// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

#include "../TridiagonalMatrixShapes.h"

using TridiagonalMatrixTraverseTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::TridiagonalMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::TridiagonalMatrix< float, TNL::Devices::Host, long >,
   TNL::Matrices::TridiagonalMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >
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

   // Row 0 has only diagonal (column 0) and superdiagonal (column 1) elements.
   view.setElement( (IndexType) 0, (IndexType) 0, (RealType) 1.0 );
   view.setElement( (IndexType) 0, (IndexType) 1, (RealType) 2.0 );

   // Rows 1-3 have all three bands.
   view.setElement( (IndexType) 1, (IndexType) 0, (RealType) 3.0 );
   view.setElement( (IndexType) 1, (IndexType) 1, (RealType) 4.0 );
   view.setElement( (IndexType) 1, (IndexType) 2, (RealType) 5.0 );

   view.setElement( (IndexType) 2, (IndexType) 1, (RealType) 6.0 );
   view.setElement( (IndexType) 2, (IndexType) 2, (RealType) 7.0 );
   view.setElement( (IndexType) 2, (IndexType) 3, (RealType) 8.0 );

   view.setElement( (IndexType) 3, (IndexType) 2, (RealType) 9.0 );
   view.setElement( (IndexType) 3, (IndexType) 3, (RealType) 10.0 );
   view.setElement( (IndexType) 3, (IndexType) 4, (RealType) 11.0 );

   // Row 4 has only subdiagonal (column 3) and diagonal (column 4) elements.
   view.setElement( (IndexType) 4, (IndexType) 3, (RealType) 12.0 );
   view.setElement( (IndexType) 4, (IndexType) 4, (RealType) 13.0 );
}

template< typename MatrixType >
void
test_forElements_Range()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   // Process rows 1 to 4 (i.e., rows 1, 2, 3).
   TNL::Matrices::forElements(
      view,
      (IndexType) 1,
      (IndexType) 4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) mutable
      {
         // For TridiagonalMatrix, columnIdx is the actual column index, not just localIdx. A wrong column
         // index spoils the row sum, which is checked on the host.
         const bool validColumn = columnIdx == rowIdx + localIdx - 1;
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], validColumn ? value : 1000 );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forElements(
      constView,
      (IndexType) 1,
      (IndexType) 4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename MatrixType >
void
test_forAllElements()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType totalSum( 1, 0 );
   auto totalSumView = totalSum.getView();

   TNL::Matrices::forAllElements(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( totalSumView[ 0 ], value );
      } );

   EXPECT_EQ( totalSum.getElement( 0 ), 91 );

   const auto constView = view;
   totalSum = 0;

   TNL::Matrices::forAllElements(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( totalSumView[ 0 ], value );
      } );

   EXPECT_EQ( totalSum.getElement( 0 ), 91 );
}

template< typename MatrixType >
void
test_forElements_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   IndexVectorType rowIndexes{ 1, 3 };

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forElements(
      view,
      rowIndexes.getView(),
      (IndexType) 0,
      (IndexType) 2,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forElements(
      constView,
      rowIndexes.getView(),
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename MatrixType >
void
test_forElementsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   // Process even rows.
   TNL::Matrices::forElementsIf(
      view,
      (IndexType) 0,
      (IndexType) 5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forElementsIf(
      constView,
      (IndexType) 0,
      (IndexType) 5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx < 2;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename MatrixType >
void
test_forAllElementsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forAllElementsIf(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx > 1;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forAllElementsIf(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx != 2;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );
}

template< typename MatrixType >
void
test_forRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   auto f = [ = ] __cuda_callable__( RowView& row ) mutable
   {
      RealType sum = 0;
      for( IndexType i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   auto const_f = [ = ] __cuda_callable__( const ConstRowView& row ) mutable
   {
      RealType sum = 0;
      for( IndexType i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   TNL::Matrices::forRows( view, (IndexType) 1, (IndexType) 4, f );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forRows( constView, (IndexType) 1, (IndexType) 4, const_f );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   rowSums = 0;
   TNL::Matrices::forAllRows( view, f );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   rowSums = 0;
   TNL::Matrices::forAllRows( constView, const_f );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );
}

template< typename MatrixType >
void
test_forRows_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   IndexVectorType rowIndexes{ 0, 2, 4 };

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forRows(
      view,
      rowIndexes.getView(),
      (IndexType) 0,
      (IndexType) 3,
      [ = ] __cuda_callable__( RowView& row ) mutable
      {
         RealType sum = 0;
         for( IndexType i = 0; i < row.getSize(); i++ )
            sum += row.getValue( i );
         rowSumsView[ row.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forRows(
      constView,
      rowIndexes.getView(),
      [ = ] __cuda_callable__( const ConstRowView& row ) mutable
      {
         RealType sum = 0;
         for( IndexType i = 0; i < row.getSize(); i++ )
            sum += row.getValue( i );
         rowSumsView[ row.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );
}

template< typename MatrixType >
void
test_forRowsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forRowsIf(
      view,
      (IndexType) 0,
      (IndexType) 5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx % 2 == 1;
      },
      [ = ] __cuda_callable__( RowView& row ) mutable
      {
         RealType sum = 0;
         for( IndexType i = 0; i < row.getSize(); i++ )
            sum += row.getValue( i );
         rowSumsView[ row.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constView = view;
   rowSums = 0;

   TNL::Matrices::forRowsIf(
      constView,
      (IndexType) 0,
      (IndexType) 5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx < 2;
      },
      [ = ] __cuda_callable__( const ConstRowView& row ) mutable
      {
         RealType sum = 0;
         for( IndexType i = 0; i < row.getSize(); i++ )
            sum += row.getValue( i );
         rowSumsView[ row.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename MatrixType >
void
test_forAllRowsIf()
{
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forAllRowsIf(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx >= 2;
      },
      [ = ] __cuda_callable__( RowView& row ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 3 );

   const auto constView = view;
   counter = 0;

   TNL::Matrices::forAllRowsIf(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx != 2;
      },
      [ = ] __cuda_callable__( const ConstRowView& row ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 4 );
}

// Test fixture
// Checks the elements recorded by a traversal. The slot `3 * rowIdx + localIdx` holds the column index
// and the value of the visited element, or -1 and 0 if the element was not visited.
template< typename IndexVector, typename RealVector >
void
checkVisitedElements(
   const IndexVector& visitedColumns,
   const RealVector& visitedValues,
   int rows,
   int columns,
   const std::vector< bool >& processedRows )
{
   using IndexType = typename IndexVector::ValueType;
   using RealType = typename RealVector::ValueType;
   TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostColumns;
   TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType > hostValues;
   hostColumns = visitedColumns;
   hostValues = visitedValues;
   for( int rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( int localIdx = 0; localIdx < 3; localIdx++ ) {
         SCOPED_TRACE(
            "matrix " + std::to_string( rows ) + "x" + std::to_string( columns ) + ", row " + std::to_string( rowIdx )
            + ", localIdx " + std::to_string( localIdx ) );
         const int columnIdx = processedRows[ rowIdx ] ? getTridiagonalColumnIndex( rowIdx, localIdx, columns ) : -1;
         EXPECT_EQ( hostColumns[ 3 * rowIdx + localIdx ], columnIdx );
         EXPECT_EQ( hostValues[ 3 * rowIdx + localIdx ], columnIdx >= 0 ? getTridiagonalTestValue( rowIdx, columnIdx ) : 0 );
      }
}

template< typename MatrixType >
void
test_forElements_Shapes()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RealVector = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   for( const auto& shape : getTridiagonalTestShapes() ) {
      const IndexType rows = shape.first;
      const IndexType columns = shape.second;
      MatrixType matrix( rows, columns );
      setupTridiagonalTestMatrix( matrix );
      const auto constView = matrix.getConstView();

      IndexVector visitedColumns( 3 * rows );
      RealVector visitedValues( 3 * rows );
      auto visitedColumns_view = visitedColumns.getView();
      auto visitedValues_view = visitedValues.getView();
      auto reset = [ & ]()
      {
         visitedColumns.setValue( -1 );
         visitedValues.setValue( 0 );
      };
      auto record =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         visitedColumns_view[ 3 * rowIdx + localIdx ] = columnIdx;
         visitedValues_view[ 3 * rowIdx + localIdx ] = value;
      };

      // Row indexes in reverse order and the selection of rows for the conditional traversal
      std::vector< IndexType > hostRowIndexes( rows );
      for( IndexType i = 0; i < rows; i++ )
         hostRowIndexes[ i ] = rows - 1 - i;
      IndexVector rowIndexes( hostRowIndexes );
      auto evenRows = [] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx % 2 == 0;
      };
      std::vector< bool > allRows( rows, true );
      std::vector< bool > allButFirstRow( rows, true );
      allButFirstRow[ 0 ] = false;
      std::vector< bool > allButLastRow( rows, true );
      allButLastRow[ rows - 1 ] = false;
      std::vector< bool > selectedEvenRows( rows );
      for( IndexType i = 0; i < rows; i++ )
         selectedEvenRows[ i ] = i % 2 == 0;

      reset();
      TNL::Matrices::forAllElements( matrix, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, allRows );

      reset();
      TNL::Matrices::forAllElements( constView, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, allRows );

      reset();
      TNL::Matrices::forElements( matrix, (IndexType) 1, rows, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, allButFirstRow );

      reset();
      TNL::Matrices::forElements( matrix, rowIndexes, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, allRows );

      // The array is reversed, so skipping its first item skips the last row
      reset();
      TNL::Matrices::forElements( constView, rowIndexes, (IndexType) 1, rows, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, allButLastRow );

      reset();
      TNL::Matrices::forAllElementsIf( matrix, evenRows, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, selectedEvenRows );

      reset();
      TNL::Matrices::forElementsIf( constView, rowIndexes, (IndexType) 0, rows, evenRows, record );
      checkVisitedElements( visitedColumns, visitedValues, rows, columns, selectedEvenRows );
   }
}

template< typename MatrixType >
class TridiagonalMatrixTraverseTest : public ::testing::Test
{
protected:
   using MatrixType_ = MatrixType;
};

TYPED_TEST_SUITE_P( TridiagonalMatrixTraverseTest );

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forElements_Range )
{
   test_forElements_Range< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forAllElements )
{
   test_forAllElements< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forElements_WithIndexArray )
{
   test_forElements_WithIndexArray< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forElementsIf )
{
   test_forElementsIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forAllElementsIf )
{
   test_forAllElementsIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forRows )
{
   test_forRows< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forRows_WithIndexArray )
{
   test_forRows_WithIndexArray< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forRowsIf )
{
   test_forRowsIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forAllRowsIf )
{
   test_forAllRowsIf< TypeParam >();
}

TYPED_TEST_P( TridiagonalMatrixTraverseTest, forElements_Shapes )
{
   test_forElements_Shapes< TypeParam >();
}

REGISTER_TYPED_TEST_SUITE_P(
   TridiagonalMatrixTraverseTest,
   forElements_Range,
   forAllElements,
   forElements_WithIndexArray,
   forElementsIf,
   forAllElementsIf,
   forRows,
   forRows_WithIndexArray,
   forRowsIf,
   forAllRowsIf,
   forElements_Shapes );

INSTANTIATE_TYPED_TEST_SUITE_P( TridiagonalMatrix, TridiagonalMatrixTraverseTest, TridiagonalMatrixTraverseTypes );

#include "../../main.h"
