// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   #define LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE TNL::Devices::Host
#elif defined( __CUDACC__ )
   #define LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE TNL::Devices::Cuda
#elif defined( __HIP__ )
   #define LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE TNL::Devices::Hip
#endif

namespace LambdaMatrixTraverseTestNamespace {

template< typename Real, typename Index >
auto
createAntiDiagonalMatrix( Index size )
{
   using Device = LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE;

   auto rowLengths = [ = ] __cuda_callable__( Index rows, Index columns, Index rowIdx ) -> Index
   {
      return 1;
   };

   auto matrixElements =
      [ = ] __cuda_callable__( Index rows, Index columns, Index rowIdx, Index localIdx, Index & columnIdx, Real & value )
   {
      columnIdx = columns - 1 - rowIdx;
      value = (Real) ( columnIdx + 1 );
   };

   return TNL::Matrices::LambdaMatrixFactory< Real, Device, Index >::create( size, size, matrixElements, rowLengths );
}

template< typename Real, typename Index >
void
test_forElements_Range()
{
   using Device = LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forElements(
      matrix,
      (Index) 1,
      (Index) 4,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         EXPECT_EQ( columnIdx, size - 1 - rowIdx );
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::forElements(
      constMatrix,
      (Index) 1,
      (Index) 4,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         EXPECT_EQ( columnIdx, size - 1 - rowIdx );
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename Real, typename Index >
void
test_forAllElements()
{
   using Device = LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forAllElements(
      matrix,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         EXPECT_EQ( columnIdx, size - 1 - rowIdx );
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::forAllElements(
      constMatrix,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         EXPECT_EQ( columnIdx, size - 1 - rowIdx );
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );
}

template< typename Real, typename Index >
void
test_forRows()
{
   using Device = LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using MatrixType = decltype( createAntiDiagonalMatrix< Real, Index >( (Index) 5 ) );
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto f = [ = ] __cuda_callable__( RowView & row ) mutable
   {
      Real sum = 0;
      for( Index i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   auto const_f = [ = ] __cuda_callable__( const ConstRowView& row ) mutable
   {
      Real sum = 0;
      for( Index i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   TNL::Matrices::forRows( matrix, (Index) 1, (Index) 4, f );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::forRows( constMatrix, (Index) 1, (Index) 4, const_f );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename Real, typename Index >
void
test_forAllRows()
{
   using Device = LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using MatrixType = decltype( createAntiDiagonalMatrix< Real, Index >( (Index) 5 ) );
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto f = [ = ] __cuda_callable__( RowView & row ) mutable
   {
      Real sum = 0;
      for( Index i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   auto const_f = [ = ] __cuda_callable__( const ConstRowView& row ) mutable
   {
      Real sum = 0;
      for( Index i = 0; i < row.getSize(); i++ )
         sum += row.getValue( i );
      rowSumsView[ row.getRowIndex() ] = sum;
   };

   TNL::Matrices::forAllRows( matrix, f );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::forAllRows( constMatrix, const_f );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );
}

}  // namespace LambdaMatrixTraverseTestNamespace

TEST( LambdaMatrixTraverseTest, forElements_Range )
{
   using namespace LambdaMatrixTraverseTestNamespace;
   test_forElements_Range< double, int >();
   test_forElements_Range< float, long >();
}

TEST( LambdaMatrixTraverseTest, forAllElements )
{
   using namespace LambdaMatrixTraverseTestNamespace;
   test_forAllElements< double, int >();
   test_forAllElements< float, long >();
}

TEST( LambdaMatrixTraverseTest, forRows )
{
   using namespace LambdaMatrixTraverseTestNamespace;
   test_forRows< double, int >();
   test_forRows< float, long >();
}

TEST( LambdaMatrixTraverseTest, forAllRows )
{
   using namespace LambdaMatrixTraverseTestNamespace;
   test_forAllRows< double, int >();
   test_forAllRows< float, long >();
}

#undef LAMBDA_MATRIX_TRAVERSE_TEST_DEVICE
