// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

// Types for which MultidiagonalMatrixTraverseTest exercises the free traversal functions.
using MultidiagonalMatrixTraverseTypes = ::testing::Types<
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
test_forElements_Range()
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

   TNL::Matrices::forElements(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forElements(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
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
test_forAllElements()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType total( 1 );
   total.setValue( 0 );
   auto totalView = total.getView();

   TNL::Matrices::forAllElements(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( totalView[ 0 ], value );
      } );

   EXPECT_EQ( total.getElement( 0 ), 648 );

   const auto constView = view;
   total.setValue( 0 );

   TNL::Matrices::forAllElements(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( totalView[ 0 ], value );
      } );

   EXPECT_EQ( total.getElement( 0 ), 648 );
}

template< typename MatrixType >
void
test_forElements_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   IndexVectorType rowIndexes{ 1, 4, 2 };

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forElements(
      view,
      rowIndexes,
      0,
      3,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forElements(
      constView,
      rowIndexes,
      0,
      3,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_forElementsIf()
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

   TNL::Matrices::forElementsIf(
      view,
      0,
      6,
      [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         return rowIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forElementsIf(
      constView,
      0,
      6,
      [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         return rowIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( rowSumsView[ rowIdx ], value );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_forRows_Range()
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

   TNL::Matrices::forRows(
      view,
      1,
      4,
      [ = ] __cuda_callable__( typename MatrixType::RowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forRows(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( typename MatrixType::ConstRowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
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
test_forAllRows()
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

   TNL::Matrices::forAllRows(
      view,
      [ = ] __cuda_callable__( typename MatrixType::RowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forAllRows(
      constView,
      [ = ] __cuda_callable__( typename MatrixType::ConstRowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
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
test_forRows_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   IndexVectorType rowIndexes{ 0, 2, 4 };

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forRows(
      view,
      rowIndexes,
      0,
      3,
      [ = ] __cuda_callable__( typename MatrixType::RowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forRows(
      constView,
      rowIndexes,
      0,
      3,
      [ = ] __cuda_callable__( typename MatrixType::ConstRowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_forRowsIf()
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

   TNL::Matrices::forRowsIf(
      view,
      0,
      6,
      [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         return rowIdx % 2 == 1;
      },
      [ = ] __cuda_callable__( typename MatrixType::RowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::forRowsIf(
      constView,
      0,
      6,
      [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         return rowIdx % 2 == 1;
      },
      [ = ] __cuda_callable__( typename MatrixType::ConstRowView & rowView ) mutable
      {
         RealType sum = 0;
         for( IndexType localIdx = 0; localIdx < rowView.getSize(); localIdx++ )
            sum += rowView.getValue( localIdx );
         rowSumsView[ rowView.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );
}
