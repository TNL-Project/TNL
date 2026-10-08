// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <TNL/Functional.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

#include "../MultidiagonalMatrixShapes.h"

// Types for which MultidiagonalMatrixReduceTest exercises the free reduction functions.
using MultidiagonalMatrixReduceTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >
#elif defined( __CUDACC__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long >
#endif
   >;

template< typename MatrixType >
void
setupTestMatrix( MatrixType& matrix )
{
   using IndexType = typename MatrixType::IndexType;
   using RealType = typename MatrixType::RealType;

   const IndexType rows = 6;
   const IndexType cols = 6;
   TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > offsets{ -2, -1, 0, 1, 2 };
   matrix.setDimensions( rows, cols, offsets );

   for( IndexType row = 0; row < rows; row++ ) {
      for( IndexType localIdx = 0; localIdx < 5; localIdx++ ) {
         const IndexType column = row + ( localIdx - 2 );
         if( column >= 0 && column < cols )
            matrix.setElement( row, column, (RealType) ( row * 10 + localIdx ) );
      }
   }
}

template< typename MatrixType >
void
test_reduceRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceRows(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceRows(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_reduceRows_AutoIdentity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceRows(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceRows(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_reduceAllRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceAllRows(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceAllRows(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );
}

template< typename MatrixType >
void
test_reduceAllRows_AutoIdentity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceAllRows(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceAllRows(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );
}

template< typename MatrixType >
void
test_reduceRowsWithArgument()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );

   VectorType maxValues( matrix.getRows(), 0 );
   IndexVectorType maxColumns( matrix.getRows(), -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto reduce = [] __cuda_callable__( RealType& a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__(
                   IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool emptyRow ) mutable
   {
      maxValuesView[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumnsView[ rowIdx ] = columnIdx;
   };
   auto storeWithIdx = [ = ] __cuda_callable__(
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

   // reduceAllRowsWithArgument
   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, store, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 4 );
   EXPECT_EQ( maxColumns.getElement( 0 ), 2 );
   EXPECT_EQ( maxValues.getElement( 5 ), 52 );
   EXPECT_EQ( maxColumns.getElement( 5 ), 5 );

   // reduceRowsWithArgument (range)
   auto view = matrix.getView();
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument( view, (IndexType) 1, (IndexType) 4, fetch, reduce, store, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 1 ), 14 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 34 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 5 );
   EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped

   // reduceRowsWithArgument (array)
   IndexVectorType rowIndexes{ 0, 2, 5 };
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument( matrix, rowIndexes, fetch, reduce, storeWithIdx, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 4 );
   EXPECT_EQ( maxColumns.getElement( 0 ), 2 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 2 ), 24 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 4 );
   EXPECT_EQ( maxValues.getElement( 5 ), 52 );
   EXPECT_EQ( maxColumns.getElement( 5 ), 5 );
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

   MatrixType matrix;
   setupTestMatrix( matrix );

   VectorType maxValues( matrix.getRows(), 0 );
   IndexVectorType maxColumns( matrix.getRows(), -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) -> RealType
   {
      return value;
   };
   auto reduce = [] __cuda_callable__( RealType& a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
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

   // reduceAllRowsWithArgumentIf (range)
   TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, reduce, store, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 2 ), 24 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 4 );
   EXPECT_EQ( maxValues.getElement( 5 ), 52 );
   EXPECT_EQ( maxColumns.getElement( 5 ), 5 );

   // reduceRowsWithArgumentIf (array)
   auto view2 = matrix.getView();
   IndexVectorType rowIndexes{ 0, 2, 3, 5 };
   auto conditionArray = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgumentIf(
      view2, rowIndexes, (IndexType) 0, rowIndexes.getSize(), conditionArray, fetch, reduce, store, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition (rowIdx=0)
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 2 ), 24 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 4 );
   EXPECT_EQ( maxValues.getElement( 3 ), 34 );  // processed
   EXPECT_EQ( maxColumns.getElement( 3 ), 5 );
   EXPECT_EQ( maxValues.getElement( 5 ), 52 );  // processed
   EXPECT_EQ( maxColumns.getElement( 5 ), 5 );
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

   const IndexType diagonals = getMultidiagonalTestOffsets().size();
   for( const auto& shape : getMultidiagonalTestShapes() ) {
      const IndexType rows = shape.first;
      const IndexType columns = shape.second;
      SCOPED_TRACE( "matrix " + std::to_string( rows ) + "x" + std::to_string( columns ) );
      MatrixType matrix;
      setupMultidiagonalTestMatrix( matrix, rows, columns );
      const auto constView = matrix.getConstView();

      // Expected results: the sum of `1000 * value + columnIdx` over the row checks both the values
      // and the column indexes, the maximum is the element with the largest column index.
      std::vector< RealType > expectedSums( rows, 0 );
      std::vector< IndexType > expectedMaxColumns( rows, -1 );
      std::vector< IndexType > expectedMaxLocalIdxs( rows, -1 );
      for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
         for( IndexType localIdx = 0; localIdx < diagonals; localIdx++ ) {
            const int columnIdx = getMultidiagonalColumnIndex( rowIdx, localIdx, columns );
            if( columnIdx >= 0 ) {
               expectedSums[ rowIdx ] += 1000 * getMultidiagonalTestValue( rowIdx, columnIdx ) + columnIdx;
               expectedMaxColumns[ rowIdx ] = columnIdx;
               expectedMaxLocalIdxs[ rowIdx ] = localIdx;
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
      IndexVector emptyRows( rows );
      auto sums_view = sums.getView();
      auto maxColumns_view = maxColumns.getView();
      auto maxLocalIdxs_view = maxLocalIdxs.getView();
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
         emptyRows_view[ rowIdx ] = emptyRow;
      };
      auto checkSums = [ & ]()
      {
         HostRealVector hostSums;
         hostSums = sums;
         for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
            EXPECT_EQ( hostSums[ rowIdx ], expectedSums[ rowIdx ] ) << "row " << rowIdx;
      };

      sums.setValue( -1 );
      TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, store );
      checkSums();

      sums.setValue( -1 );
      TNL::Matrices::reduceRows( constView, rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes );
      checkSums();

      maxColumns.setValue( -2 );
      maxLocalIdxs.setValue( -2 );
      emptyRows.setValue( -1 );
      TNL::Matrices::reduceAllRowsWithArgument( constView, valueFetch, TNL::MaxWithArg{}, storeWithArgument );
      HostIndexVector hostColumns;
      HostIndexVector hostLocalIdxs;
      HostIndexVector hostEmptyRows;
      hostColumns = maxColumns;
      hostLocalIdxs = maxLocalIdxs;
      hostEmptyRows = emptyRows;
      for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ ) {
         EXPECT_EQ( hostEmptyRows[ rowIdx ], expectedMaxColumns[ rowIdx ] < 0 ? 1 : 0 ) << "row " << rowIdx;
         EXPECT_EQ( hostColumns[ rowIdx ], expectedMaxColumns[ rowIdx ] ) << "row " << rowIdx;
         EXPECT_EQ( hostLocalIdxs[ rowIdx ], expectedMaxLocalIdxs[ rowIdx ] ) << "row " << rowIdx;
      }
   }
}

// Test fixture
template< typename MatrixType >
class MultidiagonalMatrixReduceTest : public ::testing::Test
{
protected:
   using MatrixType_ = MatrixType;
};

TYPED_TEST_SUITE_P( MultidiagonalMatrixReduceTest );

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceRows )
{
   test_reduceRows< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceRows_AutoIdentity )
{
   test_reduceRows_AutoIdentity< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceAllRows )
{
   test_reduceAllRows< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceAllRows_AutoIdentity )
{
   test_reduceAllRows_AutoIdentity< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceRowsWithArgument )
{
   test_reduceRowsWithArgument< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceRowsWithArgumentIf )
{
   test_reduceRowsWithArgumentIf< TypeParam >();
}

TYPED_TEST_P( MultidiagonalMatrixReduceTest, reduceRows_Shapes )
{
   test_reduceRows_Shapes< TypeParam >();
}

REGISTER_TYPED_TEST_SUITE_P(
   MultidiagonalMatrixReduceTest,
   reduceRows,
   reduceRows_AutoIdentity,
   reduceAllRows,
   reduceAllRows_AutoIdentity,
   reduceRowsWithArgument,
   reduceRowsWithArgumentIf,
   reduceRows_Shapes );

INSTANTIATE_TYPED_TEST_SUITE_P( MultidiagonalMatrix, MultidiagonalMatrixReduceTest, MultidiagonalMatrixReduceTypes );

#include "../../main.h"
