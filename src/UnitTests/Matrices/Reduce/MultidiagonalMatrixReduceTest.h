// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functional.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

// Types for which MultidiagonalMatrixReduceTest exercises the free reduction functions.
using MultidiagonalMatrixReduceTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long >
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
   auto reduce = [] __cuda_callable__( RealType & a, const RealType& b, IndexType& aIdx, IndexType bIdx )
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
   auto reduce = [] __cuda_callable__( RealType & a, const RealType& b, IndexType& aIdx, IndexType bIdx )
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
   auto conditionArray = [] __cuda_callable__( IndexType idx ) -> bool
   {
      return idx >= 1;
   };
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgumentIf(
      view2, rowIndexes, (IndexType) 0, rowIndexes.getSize(), conditionArray, fetch, reduce, store, (RealType) 0 );
   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition (idx=0)
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 2 ), 24 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 4 );
   EXPECT_EQ( maxValues.getElement( 3 ), 34 );  // processed
   EXPECT_EQ( maxColumns.getElement( 3 ), 5 );
   EXPECT_EQ( maxValues.getElement( 5 ), 52 );  // processed
   EXPECT_EQ( maxColumns.getElement( 5 ), 5 );
}
