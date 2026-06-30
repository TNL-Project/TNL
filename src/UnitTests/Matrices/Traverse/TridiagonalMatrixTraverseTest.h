// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   #define TRIDIAGONAL_MATRIX_TEST_DEVICE TNL::Devices::Host
#elif defined( __CUDACC__ )
   #define TRIDIAGONAL_MATRIX_TEST_DEVICE TNL::Devices::Cuda
#elif defined( __HIP__ )
   #define TRIDIAGONAL_MATRIX_TEST_DEVICE TNL::Devices::Hip
#endif

namespace TridiagonalMatrixTraverseTestNamespace {

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
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
      {
         // For TridiagonalMatrix, columnIdx is the actual column index, not just localIdx.
         EXPECT_GE( columnIdx, rowIdx - 1 );
         EXPECT_LE( columnIdx, rowIdx + 1 );
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowSumsView[ rowIdx ], value );
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
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
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
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
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
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
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
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType & value ) mutable
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
      [ = ] __cuda_callable__( RowView & row ) mutable
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
      [ = ] __cuda_callable__( RowView & row ) mutable
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
      [ = ] __cuda_callable__( RowView & row ) mutable
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

}  // namespace TridiagonalMatrixTraverseTestNamespace

TEST( TridiagonalMatrixTraverseTest, forElements_Range )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forElements_Range< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forElements_Range< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forAllElements )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forAllElements< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forAllElements< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forElements_WithIndexArray )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forElements_WithIndexArray< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forElements_WithIndexArray< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forElementsIf )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forElementsIf< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forElementsIf< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forAllElementsIf )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forAllElementsIf< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forAllElementsIf< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forRows )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forRows< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forRows< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forRows_WithIndexArray )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forRows_WithIndexArray< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forRows_WithIndexArray< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forRowsIf )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forRowsIf< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forRowsIf< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixTraverseTest, forAllRowsIf )
{
   using namespace TridiagonalMatrixTraverseTestNamespace;
   test_forAllRowsIf< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_TEST_DEVICE, int > >();
   test_forAllRowsIf< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_TEST_DEVICE, long > >();
}

#undef TRIDIAGONAL_MATRIX_TEST_DEVICE
