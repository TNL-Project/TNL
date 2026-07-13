// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   #define TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Host
#elif defined( __CUDACC__ )
   #define TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Cuda
#elif defined( __HIP__ )
   #define TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Hip
#endif

namespace TridiagonalMatrixReduceTestNamespace {

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
      // For TridiagonalMatrix, the second fetch argument is the actual column index.
      EXPECT_GE( columnIdx, row - 1 );
      EXPECT_LE( columnIdx, row + 1 );
      return value;
   };
   auto reduce = [] __cuda_callable__( RealType & sum, const RealType& value ) -> RealType
   {
      return sum + value;
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   TNL::Matrices::reduceRows( matrix, (IndexType) 1, (IndexType) 4, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   const auto constMatrix( matrix );
   rowSums = 0;

   TNL::Matrices::reduceRows( constMatrix, (IndexType) 1, (IndexType) 4, fetch, reduce, keep, (RealType) 0 );

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
   auto reduce = [] __cuda_callable__( RealType & sum, const RealType& value ) -> RealType
   {
      return sum + value;
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   TNL::Matrices::reduceAllRows( matrix, fetch, reduce, keep, (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 3 );
   EXPECT_EQ( rowSums.getElement( 1 ), 12 );
   EXPECT_EQ( rowSums.getElement( 2 ), 21 );
   EXPECT_EQ( rowSums.getElement( 3 ), 30 );
   EXPECT_EQ( rowSums.getElement( 4 ), 25 );

   const auto constMatrix( matrix );
   rowSums = 0;

   TNL::Matrices::reduceAllRows( constMatrix, fetch, reduce, keep, (RealType) 0 );

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

}  // namespace TridiagonalMatrixReduceTestNamespace

TEST( TridiagonalMatrixReduceTest, reduceRows )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   test_reduceRows< TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int > >();
   test_reduceRows< TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixReduceTest, reduceAllRows_explicit_identity )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   test_reduceAllRows_explicit_identity<
      TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int > >();
   test_reduceAllRows_explicit_identity<
      TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixReduceTest, reduceAllRows_deduced_identity )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   test_reduceAllRows_deduced_identity<
      TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int > >();
   test_reduceAllRows_deduced_identity<
      TNL::Matrices::TridiagonalMatrix< float, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, long > >();
}

TEST( TridiagonalMatrixReduceTest, reduceRowsIf )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int >;
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( 5, 0 );
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

TEST( TridiagonalMatrixReduceTest, reduceRowsWithArgument_range )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int >;
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( 5, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( 5, -1 );
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

   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, store, (RealType) 0 );

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
   TNL::Matrices::reduceRowsWithArgument( constMatrix, (IndexType) 1, (IndexType) 4, fetch, reduce, store, (RealType) 0 );

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

TEST( TridiagonalMatrixReduceTest, reduceRowsWithArgument_array )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int >;
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( 5, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( 5, -1 );
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

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 1, 2, 4 };
   TNL::Matrices::reduceRowsWithArgument( matrix, rowIndexes, fetch, reduce, store, (RealType) 0 );

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
   TNL::Matrices::reduceRowsWithArgument( constMatrix, rowIndexes, fetch, reduce, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 1 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 2 );
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
}

TEST( TridiagonalMatrixReduceTest, reduceRowsWithArgumentIf )
{
   using namespace TridiagonalMatrixReduceTestNamespace;
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE, int >;
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;

   MatrixType matrix( 5, 5 );
   setupTestMatrix( matrix );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > maxValues( 5, 0 );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > maxColumns( 5, -1 );
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

   // Range variant
   TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, reduce, store, (RealType) 0 );

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
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowIndexes{ 0, 2, 3, 4 };
   auto conditionArray = [] __cuda_callable__( IndexType rowIdx ) -> bool
   {
      return rowIdx >= 2;
   };
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgumentIf(
      constMatrix, rowIndexes, (IndexType) 0, rowIndexes.getSize(), conditionArray, fetch, reduce, store, (RealType) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition (rowIdx=0)
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 2 ), 8 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 3 );
   EXPECT_EQ( maxValues.getElement( 3 ), 11 );  // processed
   EXPECT_EQ( maxColumns.getElement( 3 ), 4 );
   EXPECT_EQ( maxValues.getElement( 4 ), 13 );  // processed
   EXPECT_EQ( maxColumns.getElement( 4 ), 4 );
}

#undef TRIDIAGONAL_MATRIX_REDUCE_TEST_DEVICE
