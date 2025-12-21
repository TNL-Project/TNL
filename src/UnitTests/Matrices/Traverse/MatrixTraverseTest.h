// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <gtest/gtest.h>

template< typename MatrixType >
void
test_forElements_Range()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   // Create a test matrix
   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 2 ) );
   auto view = matrix.getView();

   // Set some elements
   view.setElement( 0, 1, 1.0 );
   view.setElement( 0, 3, 2.0 );
   view.setElement( 1, 2, 3.0 );
   view.setElement( 2, 1, 4.0 );
   view.setElement( 2, 4, 5.0 );
   view.setElement( 3, 0, 6.0 );
   view.setElement( 4, 3, 7.0 );

   VectorType columnSum( 5, 0 );
   auto columnSumView = columnSum.getView();

   // Test forElements with range (rows 1 to 4) - non-const version
   TNL::Matrices::forElements(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( columnSumView[ columnIdx ], value );
      } );

   // Rows 1-3 should have been processed
   EXPECT_EQ( columnSum.getElement( 0 ), 6 );  // row 3, col 0
   EXPECT_EQ( columnSum.getElement( 1 ), 4 );  // row 2, col 1
   EXPECT_EQ( columnSum.getElement( 2 ), 3 );  // row 1, col 2
   EXPECT_EQ( columnSum.getElement( 4 ), 5 );  // row 2, col 4

   const auto constView = view;
   columnSum = 0;
   // Test const version with 4-argument lambda
   TNL::Matrices::forElements(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( columnSumView[ columnIdx ], value );
      } );

   EXPECT_EQ( columnSum.getElement( 0 ), 6 );  // row 3, col 0
   EXPECT_EQ( columnSum.getElement( 1 ), 4 );  // row 2, col 1
   EXPECT_EQ( columnSum.getElement( 2 ), 3 );  // row 1, col 2
   EXPECT_EQ( columnSum.getElement( 4 ), 5 );  // row 2, col 4
}

template< typename MatrixType >
void
test_forAllElements()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   MatrixType matrix( 3, 3 );
   matrix.setRowCapacities( IndexVectorType( 3, 2 ) );
   auto view = matrix.getView();

   view.setElement( 0, 0, 1.0 );
   view.setElement( 0, 2, 2.0 );
   view.setElement( 1, 1, 3.0 );
   view.setElement( 2, 0, 4.0 );
   view.setElement( 2, 2, 5.0 );

   VectorType sum( 1 );
   sum.setValue( 0 );
   auto sumView = sum.getView();

   TNL::Matrices::forAllElements(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( sumView[ 0 ], value );
      } );
   EXPECT_EQ( sum.getElement( 0 ), 15 );  // 1+2+3+4+5

   const auto constView = view;

   IndexVectorType counter( 1 );
   counter.setValue( 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forAllElements(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex && value != 0 )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );
   EXPECT_EQ( counter.getElement( 0 ), 5 );
}

template< typename MatrixType >
void
test_forElements_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 1, 2, 2.0 );
   view.setElement( 2, 3, 3.0 );
   view.setElement( 3, 4, 4.0 );
   view.setElement( 4, 0, 5.0 );

   // Only process rows 1, 3
   IndexVectorType rowIndexes{ 1, 3 };

   VectorType result( 50, 0 );
   auto resultView = result.getView();
   auto rowIndexesView = rowIndexes.getView();

   TNL::Matrices::forElements(
      view,
      rowIndexesView,
      0,
      2,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            resultView[ rowIdx * 10 + columnIdx ] = value;
      } );

   EXPECT_EQ( result.getElement( 12 ), 2 );  // row 1, col 2
   EXPECT_EQ( result.getElement( 34 ), 4 );  // row 3, col 4

   const auto constView = view;
   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forElements(
      constView,
      rowIndexesView,
      0,
      2,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex && value != 0 )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

template< typename MatrixType >
void
test_forElements_WithFullIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   MatrixType matrix( 4, 4 );
   matrix.setRowCapacities( IndexVectorType( 4, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 2, 3, 2.0 );

   IndexVectorType rowIndexes{ 0, 2 };

   VectorType sum( 1, 0 );
   auto sumView = sum.getView();

   TNL::Matrices::forElements(
      view,
      rowIndexes.getView(),
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( sumView[ 0 ], value );
      } );
   EXPECT_EQ( sum.getElement( 0 ), 3 );  // 1 + 2

   const auto constView = view;
   rowIndexes = { 1, 2 };
   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forElements(
      constView,
      rowIndexes.getView(),
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex && value != 0 )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 1 );
}

template< typename MatrixType >
void
test_forElementsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 1, 2, 2.0 );
   view.setElement( 2, 3, 3.0 );
   view.setElement( 3, 4, 4.0 );
   view.setElement( 4, 0, 5.0 );

   VectorType sum( 1 );
   sum.setValue( 0 );
   auto sumView = sum.getView();

   // Only process even rows
   TNL::Matrices::forElementsIf(
      view,
      0,
      5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( sumView[ 0 ], value );
      } );

   EXPECT_EQ( sum.getElement( 0 ), 9 );  // rows 0, 2, 4: 1+3+5
   const auto constView = view;
   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   // Only process rows >= 2
   TNL::Matrices::forElementsIf(
      constView,
      0,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx >= 2;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex && value != 0 )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );
   EXPECT_EQ( counter.getElement( 0 ), 2 );  // rows 2, 3
}

template< typename MatrixType >
void
test_forAllElementsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   constexpr IndexType paddingIndex = TNL::Matrices::paddingIndex< IndexType >;

   MatrixType matrix( 4, 4 );
   matrix.setRowCapacities( IndexVectorType( 4, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 1, 2, 2.0 );
   view.setElement( 2, 3, 3.0 );
   view.setElement( 3, 0, 4.0 );

   VectorType sum( 1 );
   sum.setValue( 0 );
   auto sumView = sum.getView();

   // Only process rows > 1
   TNL::Matrices::forAllElementsIf(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx > 1;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIdx, RealType & value ) mutable
      {
         if( columnIdx != paddingIndex )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( sumView[ 0 ], value );
      } );

   EXPECT_EQ( sum.getElement( 0 ), 7 );  // rows 2, 3: 3+4

   const auto constView = view;
   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   // Only process rows != 1
   TNL::Matrices::forAllElementsIf(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx != 1;
      },
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable
      {
         if( columnIdx != paddingIndex && value != 0 )
            TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 3 );  // rows 0, 2, 3
}

template< typename MatrixType >
void
test_forRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 2 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 1, 2, 2.0 );
   view.setElement( 1, 3, 3.0 );
   view.setElement( 2, 4, 4.0 );
   view.setElement( 3, 0, 5.0 );
   view.setElement( 4, 1, 6.0 );

   const auto constView = view;
   IndexVectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   auto f = [ = ] __cuda_callable__( RowView & row ) mutable
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

   TNL::Matrices::forRows( view, 1, 4, f );
   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // not processed
   EXPECT_EQ( rowSums.getElement( 1 ), 5 );  // 2 + 3
   EXPECT_EQ( rowSums.getElement( 2 ), 4 );  // 4
   EXPECT_EQ( rowSums.getElement( 3 ), 5 );  // 5
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );  // not processed

   rowSums = 0;
   TNL::Matrices::forRows( constView, 1, 4, const_f );
   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // not processed
   EXPECT_EQ( rowSums.getElement( 1 ), 5 );  // 2 + 3
   EXPECT_EQ( rowSums.getElement( 2 ), 4 );  // 4
   EXPECT_EQ( rowSums.getElement( 3 ), 5 );  // 5
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );  // not processed

   rowSums = 0;
   TNL::Matrices::forAllRows( view, f );
   EXPECT_EQ( rowSums.getElement( 0 ), 1 );  // 1
   EXPECT_EQ( rowSums.getElement( 1 ), 5 );  // 2 + 3
   EXPECT_EQ( rowSums.getElement( 2 ), 4 );  // 4
   EXPECT_EQ( rowSums.getElement( 3 ), 5 );  // 5
   EXPECT_EQ( rowSums.getElement( 4 ), 6 );  // 6

   rowSums = 0;
   TNL::Matrices::forAllRows( constView, const_f );
   EXPECT_EQ( rowSums.getElement( 0 ), 1 );  // 1
   EXPECT_EQ( rowSums.getElement( 1 ), 5 );  // 2 + 3
   EXPECT_EQ( rowSums.getElement( 2 ), 4 );  // 4
   EXPECT_EQ( rowSums.getElement( 3 ), 5 );  // 5
   EXPECT_EQ( rowSums.getElement( 4 ), 6 );  // 6
}

template< typename MatrixType >
void
test_forRows_WithIndexArray()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 1, 1.0 );
   view.setElement( 1, 2, 2.0 );
   view.setElement( 2, 3, 3.0 );
   view.setElement( 3, 4, 4.0 );
   view.setElement( 4, 0, 5.0 );

   IndexVectorType rowIndexes{ 0, 2, 4 };

   IndexVectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forRows( view,
                           rowIndexes.getView(),
                           0,
                           3,
                           [ = ] __cuda_callable__( RowView & row ) mutable
                           {
                              RealType sum = 0;
                              for( IndexType i = 0; i < row.getSize(); i++ )
                                 sum += row.getValue( i );
                              rowSumsView[ row.getRowIndex() ] = sum;
                           } );

   EXPECT_EQ( rowSums.getElement( 0 ), 1 );
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // not processed
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // not processed
   EXPECT_EQ( rowSums.getElement( 4 ), 5 );

   const auto constView = view.getConstView();

   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forRows( constView,
                           rowIndexes.getView(),
                           0,
                           2,
                           [ = ] __cuda_callable__( const ConstRowView& row ) mutable
                           {
                              TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ],
                                                                                                         (IndexType) 1 );
                           } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

template< typename MatrixType >
void
test_forRows_WithFullIndexArray()
{
   using IndexType = typename MatrixType::IndexType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 4, 4 );
   matrix.setRowCapacities( IndexVectorType( 4, 1 ) );
   auto view = matrix.getView();

   view.setElement( 1, 1, 1.0 );
   view.setElement( 2, 1, 1.0 );
   view.setElement( 3, 3, 2.0 );

   IndexVectorType rowIndexes{ 1, 3 };

   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   TNL::Matrices::forRows( view,
                           rowIndexes.getView(),
                           [ = ] __cuda_callable__( RowView & row ) mutable
                           {
                              TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ],
                                                                                                         (IndexType) 1 );
                           } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );

   const auto constView = view;
   counter = 0;
   TNL::Matrices::forRows( constView,
                           rowIndexes.getView(),
                           [ = ] __cuda_callable__( const ConstRowView& row ) mutable
                           {
                              TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ],
                                                                                                         (IndexType) 1 );
                           } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

template< typename MatrixType >
void
test_forRowsIf()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 5, 5 );
   matrix.setRowCapacities( IndexVectorType( 5, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 0, 1.0 );
   view.setElement( 1, 1, 2.0 );
   view.setElement( 2, 2, 3.0 );
   view.setElement( 3, 3, 4.0 );
   view.setElement( 4, 4, 5.0 );

   IndexVectorType rowSums( 5, 0 );
   auto rowSumsView = rowSums.getView();

   // Only process odd-indexed rows
   TNL::Matrices::forRowsIf(
      view,
      0,
      5,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return ( rowIdx % 2 ) == 1;
      },
      [ = ] __cuda_callable__( RowView & row ) mutable
      {
         RealType sum = 0;
         for( IndexType i = 0; i < row.getSize(); i++ )
            sum += row.getValue( i );
         rowSumsView[ row.getRowIndex() ] = sum;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 2 );
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   EXPECT_EQ( rowSums.getElement( 3 ), 4 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constView = view;
   IndexVectorType counter( 1, 0 );
   auto counterView = counter.getView();

   // Only process rows < 2
   TNL::Matrices::forRowsIf(
      constView,
      0,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx < 2;
      },
      [ = ] __cuda_callable__( const ConstRowView& row ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

template< typename MatrixType >
void
test_forAllRowsIf()
{
   using IndexType = typename MatrixType::IndexType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, typename MatrixType::DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   MatrixType matrix( 4, 4 );
   matrix.setRowCapacities( IndexVectorType( 4, 1 ) );
   auto view = matrix.getView();

   view.setElement( 0, 0, 1.0 );
   view.setElement( 1, 1, 2.0 );
   view.setElement( 2, 2, 3.0 );
   view.setElement( 3, 3, 4.0 );

   IndexVectorType counter( 1 );
   counter.setValue( 0 );
   auto counterView = counter.getView();

   // Only process rows >= 2
   TNL::Matrices::forAllRowsIf(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx >= 2;
      },
      [ = ] __cuda_callable__( RowView & row ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 2 );

   const auto constView = view;
   counter = 0;
   // Only process rows != 1
   TNL::Matrices::forAllRowsIf(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx ) -> bool
      {
         return rowIdx != 1;
      },
      [ = ] __cuda_callable__( const ConstRowView& row ) mutable
      {
         TNL::Algorithms::AtomicOperations< typename MatrixType::DeviceType >::add( counterView[ 0 ], (IndexType) 1 );
      } );

   EXPECT_EQ( counter.getElement( 0 ), 3 );
}

// Test fixture for typed tests
template< typename Matrix >
class MatrixTraverseTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;
};

TYPED_TEST_SUITE_P( MatrixTraverseTest );

TYPED_TEST_P( MatrixTraverseTest, forElements_RangeTest )
{
   test_forElements_Range< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forAllElementsTest )
{
   test_forAllElements< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forElements_WithIndexArrayTest )
{
   test_forElements_WithIndexArray< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forElements_WithFullIndexArrayTest )
{
   test_forElements_WithFullIndexArray< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forElementsIfTest )
{
   test_forElementsIf< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forAllElementsIfTest )
{
   test_forAllElementsIf< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forRowsTest )
{
   test_forRows< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forRows_WithIndexArrayTest )
{
   test_forRows_WithIndexArray< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forRows_WithFullIndexArrayTest )
{
   test_forRows_WithFullIndexArray< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forRowsIfTest )
{
   test_forRowsIf< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( MatrixTraverseTest, forAllRowsIfTest )
{
   test_forAllRowsIf< typename TestFixture::MatrixType >();
}

REGISTER_TYPED_TEST_SUITE_P( MatrixTraverseTest,
                             forElements_RangeTest,
                             forAllElementsTest,
                             forElements_WithIndexArrayTest,
                             forElements_WithFullIndexArrayTest,
                             forElementsIfTest,
                             forAllElementsIfTest,
                             forRowsTest,
                             forRows_WithIndexArrayTest,
                             forRows_WithFullIndexArrayTest,
                             forRowsIfTest,
                             forAllRowsIfTest );

#include "../../main.h"
