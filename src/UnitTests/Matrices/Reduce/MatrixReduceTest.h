// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <vector>

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
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
    *    |  0  0  0  0  0 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 0, 4 };
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

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Compute sum of elements in each row
   TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( rows, 0 );
   auto rowSums_view = rowSums.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value;
   };
   auto store = [ = ] __cuda_callable__( const IndexType rowIdx, const RealType value ) mutable
   {
      rowSums_view[ rowIdx ] = value;
   };
   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( matrix.getSegments() ) ) {
      SCOPED_TRACE( tag );

      TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 12 );  // 1+2+4+5
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

      const auto constMatrix( matrix );

      rowSums = 0;
      TNL::Matrices::reduceRows( constMatrix.getConstView(), 1, 4, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( rowSums.getElement( 4 ), 0 );  // skipped

      auto storeWithRowIndexes =
         [ = ] __cuda_callable__( const IndexType indexOfRowIdx, const IndexType rowIdx, const RealType value ) mutable
      {
         rowSums_view[ rowIdx ] = value;
      };

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };  // Process rows 1, 2, and 4
      rowSums = 0;
      TNL::Matrices::reduceRows( matrix, rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes, RealType( 0 ), launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

      rowSums = 0;
      TNL::Matrices::reduceRows(
         matrix.getConstView(), rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes, RealType( 0 ), launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

      // The same with the identity deduced from TNL::Plus
      rowSums = 0;
      TNL::Matrices::reduceRows( matrix, rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes, launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16

      rowSums = 0;
      TNL::Matrices::reduceRows( matrix.getConstView(), rowIndexes, fetch, TNL::Plus{}, storeWithRowIndexes, launch_config );
      EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 1 ), 13 );  // 6+7
      EXPECT_EQ( rowSums.getElement( 2 ), 27 );  // 8+9+10
      EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( rowSums.getElement( 4 ), 58 );  // 13+14+15+16
   }
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
    *    |  0  0  0  0  0 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 0, 4 };
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
   auto store = [ = ] __cuda_callable__( const IndexType indexOfRowIdx, const IndexType rowIdx, const IndexType value ) mutable
   {
      rowCounts_view[ rowIdx ] = value;
   };

   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( matrix.getSegments() ) ) {
      SCOPED_TRACE( tag );

      rowCounts = 0;
      TNL::Matrices::reduceAllRowsIf( matrix, condition, fetch, TNL::Plus{}, store, 0, launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16

      rowCounts = 0;
      const auto constMatrix( matrix );
      TNL::Matrices::reduceRowsIf( constMatrix.getConstView(), 1, 4, condition, fetch, TNL::Plus{}, store, 0, launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // skipped by condition and range
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( rowCounts.getElement( 4 ), 0 );  // skipped by range

      // Test reduceRowsIf with array of row indexes
      // rowIndexes = {1, 2, 4}, condition filters rowIdx >= 2 (so row 1 is
      // skipped, rows 2 and 4 are processed)
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };
      auto conditionArray = [] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx >= 2;
      };

      rowCounts = 0;
      TNL::Matrices::reduceRowsIf(
         matrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, TNL::Plus{}, store, 0, launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16

      // Const version
      rowCounts = 0;
      TNL::Matrices::reduceRowsIf(
         constMatrix.getConstView(),
         rowIndexes,
         0,
         rowIndexes.getSize(),
         conditionArray,
         fetch,
         TNL::Plus{},
         store,
         0,
         launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16

      // The same with the identity deduced from TNL::Plus
      rowCounts = 0;
      TNL::Matrices::reduceRowsIf(
         matrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, TNL::Plus{}, store, launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16

      rowCounts = 0;
      TNL::Matrices::reduceRowsIf(
         constMatrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, TNL::Plus{}, store, launch_config );
      EXPECT_EQ( rowCounts.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( rowCounts.getElement( 2 ), 3 );  // 8, 9, 10
      EXPECT_EQ( rowCounts.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( rowCounts.getElement( 4 ), 4 );  // 13, 14, 15, 16
   }
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
    *    |  0  0  0  0  0 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   // Row 3 has no nonzero elements. Sparse matrices report it as empty (even with padding elements),
   // dense matrices store its zeros explicitly and so the maximum is found in the column 0.
   const IndexType emptyRowColumn = TNL::Matrices::is_dense_matrix_v< MatrixType > ? 0 : -1;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 0, 4 };
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

   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );

   ////
   // Test reduceRowsWithArgument: find max element and its column index
   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( rows, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( rows, -1 );
   auto maxValues_view = maxValues.getView();
   auto maxColumns_view = maxColumns.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value;
   };
   auto reduce = [] __cuda_callable__( RealType& a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
      else if( b == a && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__(
                   const IndexType rowIdx,
                   const IndexType localIdx,
                   const IndexType columnIdx,
                   const RealType& value,
                   bool emptyRow ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumns_view[ rowIdx ] = columnIdx;
   };
   auto storeWithRowIndexes = [ = ] __cuda_callable__(
                                 const IndexType indexOfRowIdx,
                                 const IndexType rowIdx,
                                 const IndexType localIdx,
                                 const IndexType columnIdx,
                                 const RealType& value,
                                 bool emptyRow ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumns_view[ rowIdx ] = columnIdx;
   };

   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( matrix.getSegments() ) ) {
      SCOPED_TRACE( tag );

      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, store, (RealType) 0, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 5 );
      EXPECT_EQ( maxColumns.getElement( 0 ), 4 );
      EXPECT_EQ( maxValues.getElement( 1 ), 7 );
      EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( maxColumns.getElement( 3 ), emptyRowColumn );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      maxValues = 0;
      maxColumns = -1;
      const auto constMatrix( matrix );
      TNL::Matrices::reduceRowsWithArgument(
         constMatrix.getConstView(), 2, 4, fetch, reduce, store, (RealType) 0, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( maxColumns.getElement( 3 ), emptyRowColumn );
      EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 4 ), -1 );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };  // Process rows 1, 2, and 4
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgument(
         constMatrix.getConstView(), rowIndexes, fetch, reduce, storeWithRowIndexes, (RealType) 0, launch_config );

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

      // The same with the identity deduced from TNL::MaxWithArg
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgument( matrix, rowIndexes, fetch, TNL::MaxWithArg{}, storeWithRowIndexes, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( maxValues.getElement( 1 ), 7 );
      EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgument(
         constMatrix, rowIndexes, fetch, TNL::MaxWithArg{}, storeWithRowIndexes, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( maxValues.getElement( 1 ), 7 );
      EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
   }
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
    *    |  0  0  0  0  0 |
    *    \ 13 14 15  0 16 /
    */

   const IndexType rows = 5;
   const IndexType cols = 5;

   // Row 3 has no nonzero elements. Sparse matrices report it as empty (even with padding elements),
   // dense matrices store its zeros explicitly and so the maximum is found in the column 0.
   const IndexType emptyRowColumn = TNL::Matrices::is_dense_matrix_v< MatrixType > ? 0 : -1;

   MatrixType matrix( rows, cols );

   // For sparse matrices, set row capacities
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 0, 4 };
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
   auto reduce = [] __cuda_callable__( RealType& a, const RealType& b, IndexType& aIdx, IndexType bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
      else if( b == a && bIdx < aIdx ) {
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__(
                   const IndexType indexOfRowIdx,
                   const IndexType rowIdx,
                   const IndexType localIdx,
                   const IndexType columnIdx,
                   const RealType& value,
                   bool emptyRow ) mutable
   {
      maxValues_view[ rowIdx ] = value;
      if( ! emptyRow )
         maxColumns_view[ rowIdx ] = columnIdx;
   };

   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( matrix.getSegments() ) ) {
      SCOPED_TRACE( tag );

      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, reduce, store, (RealType) 0, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( maxColumns.getElement( 3 ), emptyRowColumn );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      const auto constMatrix( matrix );
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgumentIf(
         constMatrix.getConstView(), 1, 4, condition, fetch, reduce, store, (RealType) 0, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // empty row
      EXPECT_EQ( maxColumns.getElement( 3 ), emptyRowColumn );
      EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped
      EXPECT_EQ( maxColumns.getElement( 4 ), -1 );

      // Test reduceRowsWithArgumentIf with array of row indexes
      // rowIndexes = {1, 2, 4}, condition filters rowIdx >= 2 (so row 1 is skipped, rows 2 and 4 are
      // processed)
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };
      auto conditionArray = [] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx >= 2;
      };
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgumentIf(
         matrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, reduce, store, (RealType) 0, launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 3 ), -1 );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      // Const version
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgumentIf(
         constMatrix.getConstView(),
         rowIndexes,
         0,
         rowIndexes.getSize(),
         conditionArray,
         fetch,
         reduce,
         store,
         (RealType) 0,
         launch_config );

      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 3 ), -1 );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      // The same with the identity deduced from TNL::MaxWithArg
      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgumentIf(
         matrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, TNL::MaxWithArg{}, store, launch_config );
      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 3 ), -1 );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );

      maxValues = 0;
      maxColumns = -1;
      TNL::Matrices::reduceRowsWithArgumentIf(
         constMatrix, rowIndexes, 0, rowIndexes.getSize(), conditionArray, fetch, TNL::MaxWithArg{}, store, launch_config );
      EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
      EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1)
      EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
      EXPECT_EQ( maxValues.getElement( 2 ), 10 );  // max of {8, 9, 10}
      EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
      EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // not in array
      EXPECT_EQ( maxColumns.getElement( 3 ), -1 );
      EXPECT_EQ( maxValues.getElement( 4 ), 16 );  // max of {13, 14, 15, 16}
      EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
   }
}

// Tests the overloads with an interval of the array of row indexes and the conditional overloads
// with the whole array. Besides the results, the test checks the index passed to the store function,
// which is the position within the array for the interval and the rank of the processed row for the
// conditional overloads.
template< typename MatrixType >
void
test_reduceRowsWithRowIndexes()
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using RealVector = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  4  5 \
    *    |  0  6  7  0  0 |
    *    |  8  0  9 10  0 |
    *    |  0  0  0  0  0 |
    *    \ 13 14 15  0 16 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;
   MatrixType matrix( rows, cols );
   typename MatrixType::RowCapacitiesType rowCapacities{ 4, 2, 3, 0, 4 };
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
   matrix.setElement( 4, 0, 13 );
   matrix.setElement( 4, 1, 14 );
   matrix.setElement( 4, 2, 15 );
   matrix.setElement( 4, 4, 16 );
   const MatrixType& constMatrix = matrix;

   const std::vector< RealType > rowSums{ 12, 13, 27, 0, 58 };
   const std::vector< RealType > rowMaxima{ 5, 7, 10, 0, 16 };
   // Dense matrices store the zeros of row 3 explicitly and so its maximum is found in the column 0
   const std::vector< IndexType > rowMaxColumns{ 4, 2, 3, 0, 4 };

   const IndexVector rowIndexes{ 4, 0, 3, 1, 2 };
   RealVector values( rows );
   IndexVector columns( rows );
   IndexVector positions( rows );
   IndexVector emptyRows( rows );
   auto values_view = values.getView();
   auto columns_view = columns.getView();
   auto positions_view = positions.getView();
   auto emptyRows_view = emptyRows.getView();

   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> RealType
   {
      return value;
   };
   auto condition = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   auto store = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const RealType& value ) mutable
   {
      values_view[ rowIdx ] = value;
      positions_view[ rowIdx ] = indexOfRowIdx;
   };
   auto storeWithArgument = [ = ] __cuda_callable__(
                               IndexType indexOfRowIdx,
                               IndexType rowIdx,
                               IndexType localIdx,
                               IndexType columnIdx,
                               const RealType& value,
                               bool emptyRow ) mutable
   {
      values_view[ rowIdx ] = value;
      columns_view[ rowIdx ] = emptyRow ? -1 : columnIdx;
      positions_view[ rowIdx ] = indexOfRowIdx;
      emptyRows_view[ rowIdx ] = emptyRow;
   };
   const RealType lowest = std::numeric_limits< RealType >::lowest();

   auto reset = [ & ]()
   {
      values.setValue( -1 );
      columns.setValue( -2 );
      positions.setValue( -1 );
      emptyRows.setValue( -1 );
   };
   // Checks that exactly the rows in `expectedPositions` were processed and that the store function
   // received the expected index for them.
   auto check = [ & ]( const std::vector< IndexType >& expectedPositions, bool withArgument )
   {
      TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType > hostValues;
      TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostColumns;
      TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostPositions;
      TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostEmptyRows;
      hostValues = values;
      hostColumns = columns;
      hostPositions = positions;
      hostEmptyRows = emptyRows;
      for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ ) {
         const bool processed = expectedPositions[ rowIdx ] >= 0;
         EXPECT_EQ( hostPositions[ rowIdx ], expectedPositions[ rowIdx ] ) << "row " << rowIdx;
         if( ! processed )
            EXPECT_EQ( hostValues[ rowIdx ], -1 ) << "row " << rowIdx;
         else if( ! withArgument )
            EXPECT_EQ( hostValues[ rowIdx ], rowSums[ rowIdx ] ) << "row " << rowIdx;
         else if( rowIdx == 3 && ! TNL::Matrices::is_dense_matrix_v< MatrixType > ) {
            // Row 3 has no nonzero elements and sparse matrices report it as empty (even with padding elements)
            EXPECT_EQ( hostValues[ rowIdx ], lowest ) << "row " << rowIdx;
            EXPECT_EQ( hostEmptyRows[ rowIdx ], 1 ) << "row " << rowIdx;
         }
         else {
            EXPECT_EQ( hostValues[ rowIdx ], rowMaxima[ rowIdx ] ) << "row " << rowIdx;
            EXPECT_EQ( hostColumns[ rowIdx ], rowMaxColumns[ rowIdx ] ) << "row " << rowIdx;
            EXPECT_EQ( hostEmptyRows[ rowIdx ], 0 ) << "row " << rowIdx;
         }
      }
   };
   // Interval [1, 4) of rowIndexes = { 4, 0, 3, 1, 2 } contains rows 0, 3 and 1 at positions 1, 2 and 3.
   const std::vector< IndexType > intervalPositions{ 1, 3, -1, 2, -1 };
   // The condition selects rows 4, 3 and 2, which are processed in this order.
   const std::vector< IndexType > conditionRanks{ -1, -1, 2, 1, 0 };

   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( matrix.getSegments() ) ) {
      SCOPED_TRACE( tag );

      reset();
      TNL::Matrices::reduceRows( matrix, rowIndexes, 1, 4, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config );
      check( intervalPositions, false );
      reset();
      TNL::Matrices::reduceRows( constMatrix, rowIndexes, 1, 4, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config );
      check( intervalPositions, false );
      reset();
      TNL::Matrices::reduceRows( matrix, rowIndexes, 1, 4, fetch, TNL::Plus{}, store, launch_config );
      check( intervalPositions, false );
      reset();
      TNL::Matrices::reduceRows( constMatrix, rowIndexes, 1, 4, fetch, TNL::Plus{}, store, launch_config );
      check( intervalPositions, false );

      reset();
      TNL::Matrices::reduceRowsWithArgument(
         matrix, rowIndexes, 1, 4, fetch, TNL::MaxWithArg{}, storeWithArgument, lowest, launch_config );
      check( intervalPositions, true );
      reset();
      TNL::Matrices::reduceRowsWithArgument(
         constMatrix, rowIndexes, 1, 4, fetch, TNL::MaxWithArg{}, storeWithArgument, lowest, launch_config );
      check( intervalPositions, true );
      reset();
      TNL::Matrices::reduceRowsWithArgument(
         matrix, rowIndexes, 1, 4, fetch, TNL::MaxWithArg{}, storeWithArgument, launch_config );
      check( intervalPositions, true );
      reset();
      TNL::Matrices::reduceRowsWithArgument(
         constMatrix, rowIndexes, 1, 4, fetch, TNL::MaxWithArg{}, storeWithArgument, launch_config );
      check( intervalPositions, true );

      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsIf( matrix, rowIndexes, condition, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config ),
         3 );
      check( conditionRanks, false );
      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsIf(
            constMatrix, rowIndexes, condition, fetch, TNL::Plus{}, store, RealType( 0 ), launch_config ),
         3 );
      check( conditionRanks, false );
      reset();
      EXPECT_EQ( TNL::Matrices::reduceRowsIf( matrix, rowIndexes, condition, fetch, TNL::Plus{}, store, launch_config ), 3 );
      check( conditionRanks, false );
      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsIf( constMatrix, rowIndexes, condition, fetch, TNL::Plus{}, store, launch_config ), 3 );
      check( conditionRanks, false );

      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsWithArgumentIf(
            matrix, rowIndexes, condition, fetch, TNL::MaxWithArg{}, storeWithArgument, lowest, launch_config ),
         3 );
      check( conditionRanks, true );
      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsWithArgumentIf(
            constMatrix, rowIndexes, condition, fetch, TNL::MaxWithArg{}, storeWithArgument, lowest, launch_config ),
         3 );
      check( conditionRanks, true );
      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsWithArgumentIf(
            matrix, rowIndexes, condition, fetch, TNL::MaxWithArg{}, storeWithArgument, launch_config ),
         3 );
      check( conditionRanks, true );
      reset();
      EXPECT_EQ(
         TNL::Matrices::reduceRowsWithArgumentIf(
            constMatrix, rowIndexes, condition, fetch, TNL::MaxWithArg{}, storeWithArgument, launch_config ),
         3 );
      check( conditionRanks, true );
   }
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

TYPED_TEST_P( MatrixReduceTest, reduceRowsWithRowIndexesTest )
{
   test_reduceRowsWithRowIndexes< typename TestFixture::MatrixType_ >();
}

REGISTER_TYPED_TEST_SUITE_P(
   MatrixReduceTest,
   reduceRowsTest,
   reduceRowsIfTest,
   reduceRowsWithArgumentTest,
   reduceRowsWithArgumentIfTest,
   reduceRowsWithRowIndexesTest );

#include "../../main.h"
