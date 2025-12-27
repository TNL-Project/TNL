// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <gtest/gtest.h>

template< typename MatrixType >
void
test_reduceRows()
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  4  5 \
    *    |  0  6  7  0  0 |
    *    |  8  0  9 10  0 |
    *    |  0  0  0 11 12 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 2, 4 };
   matrix.setRowCapacities( rowCapacities );

   matrix.setElement( 0, 0, 1 );
   matrix.setElement( 0, 1, 2 );
   matrix.setElement( 0, 3, 4 );
   matrix.setElement( 0, 4, 5 );

   matrix.setElement( 1, 1, 6 );
   matrix.setElement( 1, 2, 7 );

   matrix.setElement( 2, 0, 8 );
   matrix.setElement( 2, 2, 9 );
   matrix.setElement( 2, 3, 10 );

   matrix.setElement( 3, 3, 11 );
   matrix.setElement( 3, 4, 12 );

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Compute sum of elements in each row
   TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( rows );
   auto rowSums_view = rowSums.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const RealType value ) mutable
   {
      rowSums_view[ rowIdx ] = value;
   };
   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, keep, RealType( 0 ) );
   EXPECT_EQ( rowSums.getElement( 0 ), 12 );  // 1+2+4+5
   EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
   EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
   EXPECT_EQ( rowSums.getElement( 3 ), 23 );  // 11+12
   EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

   const auto constMatrix( matrix );

   rowSums = 0;
   TNL::Matrices::reduceRows( constMatrix.getConstView(), 1, 4, fetch, TNL::Plus{}, keep, RealType( 0 ) );
   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
   EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
   EXPECT_EQ( rowSums.getElement( 3 ), 23 );  // 11+12
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );  // skipped

   auto keepWithRowIndexes =
      [ = ] __cuda_callable__( const IndexType indexOfRowIdx, const IndexType rowIdx, const RealType value ) mutable
   {
      rowSums_view[ rowIdx ] = value;
   };

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };  // Process rows 1, 2, and 4
   rowSums = 0;
   TNL::Matrices::reduceRows( matrix, rowIndexes, fetch, TNL::Plus{}, keepWithRowIndexes, RealType( 0 ) );
   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
   EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

   rowSums = 0;
   TNL::Matrices::reduceRows( matrix.getConstView(), rowIndexes, fetch, TNL::Plus{}, keepWithRowIndexes, RealType( 0 ) );
   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
   EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16
}

template< typename MatrixType >
void
test_reduceRowsIf()
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  4  5 \
    *    |  0  6  7  0  0 |
    *    |  8  0  9 10  0 |
    *    |  0  0  0 11 12 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 2, 4 };
   matrix.setRowCapacities( rowCapacities );

   matrix.setElement( 0, 0, 1 );
   matrix.setElement( 0, 1, 2 );
   matrix.setElement( 0, 3, 4 );
   matrix.setElement( 0, 4, 5 );

   matrix.setElement( 1, 1, 6 );
   matrix.setElement( 1, 2, 7 );

   matrix.setElement( 2, 0, 8 );
   matrix.setElement( 2, 2, 9 );
   matrix.setElement( 2, 3, 10 );

   matrix.setElement( 3, 3, 11 );
   matrix.setElement( 3, 4, 12 );

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Test reduceRowsIf: count elements > 5
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowCounts( rows, 0 );
   auto rowCounts_view = rowCounts.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return value > 5;
   };
   auto condition = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType indexOfRowIdx, const IndexType rowIdx, const IndexType value ) mutable
   {
      rowCounts_view[ rowIdx ] = value;
   };
   TNL::Matrices::reduceAllRowsIf( matrix, condition, fetch, TNL::Plus{}, keep, 0 );
   EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
   EXPECT_EQ( rowCounts.getElement( 3 ), 2 );  // 11, 12
   EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16

   rowCounts = 0;
   const auto constMatrix( matrix );
   TNL::Matrices::reduceRowsIf( constMatrix.getConstView(), 1, 4, condition, fetch, TNL::Plus{}, keep, 0 );
   EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // skipped by condition and range
   EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition
   EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
   EXPECT_EQ( rowCounts.getElement( 3 ), 2 );  // 11, 12
   EXPECT_EQ( rowCounts.getElement( 4 ), 0 );  // skipped by range
}

template< typename MatrixType >
void
test_reduceRowsWithArgument()
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  4  5 \
    *    |  0  6  7  0  0 |
    *    |  8  0  9 10  0 |
    *    |  0  0  0 11 12 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 2, 4 };
   matrix.setRowCapacities( rowCapacities );

   matrix.setElement( 0, 0, 1 );
   matrix.setElement( 0, 1, 2 );
   matrix.setElement( 0, 3, 4 );
   matrix.setElement( 0, 4, 5 );

   matrix.setElement( 1, 1, 6 );
   matrix.setElement( 1, 2, 7 );

   matrix.setElement( 2, 0, 8 );
   matrix.setElement( 2, 2, 9 );
   matrix.setElement( 2, 3, 10 );

   matrix.setElement( 3, 3, 11 );
   matrix.setElement( 3, 4, 12 );

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Test reduceRowsWithArgument: find max element and its column index
   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( rows );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( rows );
   auto maxValues_view = maxValues.getView();
   auto maxColumns_view = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value;
   };
   auto reduce = [] __cuda_callable__( RealType & a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
      else if( b == a && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };
   auto keep = [ = ] __cuda_callable__(
                  const IndexType rowIdx, const IndexType localIdx, const IndexType columnIdx, const RealType& value ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      maxColumns_view[ rowIdx ] = columnIdx;
   };
   auto keepWithRowIndexes = [ = ] __cuda_callable__( const IndexType indexOfRowIdx,
                                                      const IndexType rowIdx,
                                                      const IndexType localIdx,
                                                      const IndexType columnIdx,
                                                      const RealType& value ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      maxColumns_view[ rowIdx ] = columnIdx;
   };

   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 0 ), 4 );
   EXPECT_EQ( maxValues.getElement( 1 ), 7 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 10 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 12 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 16 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

   maxValues = 0;
   maxColumns = -1;
   const auto constMatrix( matrix );
   TNL::Matrices::reduceRowsWithArgument( constMatrix.getConstView(), 2, 4, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
   EXPECT_EQ( maxValues.getElement( 2 ), 10 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 12 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 4 ), -1 );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };  // Process rows 1, 2, and 4
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument(
      constMatrix.getConstView(), rowIndexes, fetch, reduce, keepWithRowIndexes, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 7 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 10 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 3 ), -1 );
   EXPECT_EQ( maxValues.getElement( 4 ), 16 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
}

template< typename MatrixType >
void
test_reduceRowsWithArgumentIf()
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  4  5 \
    *    |  0  6  7  0  0 |
    *    |  8  0  9 10  0 |
    *    |  0  0  0 11 12 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 2, 4 };
   matrix.setRowCapacities( rowCapacities );

   matrix.setElement( 0, 0, 1 );
   matrix.setElement( 0, 1, 2 );
   matrix.setElement( 0, 3, 4 );
   matrix.setElement( 0, 4, 5 );

   matrix.setElement( 1, 1, 6 );
   matrix.setElement( 1, 2, 7 );

   matrix.setElement( 2, 0, 8 );
   matrix.setElement( 2, 2, 9 );
   matrix.setElement( 2, 3, 10 );

   matrix.setElement( 3, 3, 11 );
   matrix.setElement( 3, 4, 12 );

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Test reduceRowsWithArgumentIf: find max element > 5 and its column index
   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( rows, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( rows, -1 );
   auto maxValues_view = maxValues.getView();
   auto maxColumns_view = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value > 5 ? value : std::numeric_limits< RealType >::lowest();
   };
   auto condition = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   auto reduce = [] __cuda_callable__( RealType & a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
      else if( b == a && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };
   auto keep = [ = ] __cuda_callable__( const IndexType indexOfRowIdx,
                                        const IndexType rowIdx,
                                        const IndexType localIdx,
                                        const IndexType columnIdx,
                                        const RealType& value ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      maxColumns_view[ rowIdx ] = columnIdx;
   };

   TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
   EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 12 );  // max of {11, 12}
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

   const auto constMatrix( matrix );
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgumentIf( constMatrix.getConstView(), 1, 4, condition, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
   EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 12 );  // max of {11, 12}
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped
   EXPECT_EQ( maxColumns.getElement( 4 ), -1 );
}

// Test fixture template
template< typename MatrixType >
class MatrixReduceTest : public ::testing::Test
{
protected:
   using MatrixType_ = MatrixType;
};

TYPED_TEST_SUITE_P( MatrixReduceTest );

TYPED_TEST_P( MatrixReduceTest, reduceRowsTest )
{
   test_reduceRows< typename TestFixture::MatrixType_ >();
}

TYPED_TEST_P( MatrixReduceTest, reduceRowsIfTest )
{
   test_reduceRowsIf< typename TestFixture::MatrixType_ >();
}

TYPED_TEST_P( MatrixReduceTest, reduceRowsWithArgumentTest )
{
   test_reduceRowsWithArgument< typename TestFixture::MatrixType_ >();
}

TYPED_TEST_P( MatrixReduceTest, reduceRowsWithArgumentIfTest )
{
   test_reduceRowsWithArgumentIf< typename TestFixture::MatrixType_ >();
}

REGISTER_TYPED_TEST_SUITE_P( MatrixReduceTest,
                             reduceRowsTest,
                             reduceRowsIfTest,
                             reduceRowsWithArgumentTest,
                             reduceRowsWithArgumentIfTest );

#include "../../main.h"
