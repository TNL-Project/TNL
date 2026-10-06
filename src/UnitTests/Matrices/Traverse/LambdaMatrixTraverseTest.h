// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <vector>

#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

template< typename Real, typename Device, typename Index >
struct LambdaMatrixTraverseTestType
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
};

using LambdaMatrixTraverseTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   LambdaMatrixTraverseTestType< double, TNL::Devices::Host, int >,
   LambdaMatrixTraverseTestType< float, TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
   LambdaMatrixTraverseTestType< double, TNL::Devices::Cuda, int >,
   LambdaMatrixTraverseTestType< float, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   LambdaMatrixTraverseTestType< double, TNL::Devices::Hip, int >,
   LambdaMatrixTraverseTestType< float, TNL::Devices::Hip, long >
#endif
   >;
namespace detail {

// Stateless functors replacing __cuda_callable__ lambdas.
// nvcc (CUDA 13.3) rejects extended __host__ __device__ lambdas inside
// functions with deduced return type, so the lambdas are lifted to
// namespace-scope functors and the helpers get explicit trailing return types.

template< typename Index >
struct AntiDiagonalRowLengths
{
   __cuda_callable__
   Index
   operator()( Index rows, Index columns, Index rowIdx ) const
   {
      return 1;
   }
};

template< typename Real, typename Index >
struct AntiDiagonalMatrixElements
{
   __cuda_callable__
   void
   operator()( Index rows, Index columns, Index rowIdx, Index localIdx, Index& columnIdx, Real& value ) const
   {
      columnIdx = columns - 1 - rowIdx;
      value = static_cast< Real >( columnIdx + 1 );
   }
};

}  // namespace detail

template< typename TestType >
auto
createAntiDiagonalMatrix( typename TestType::IndexType size ) -> TNL::Matrices::LambdaMatrix<
   detail::AntiDiagonalMatrixElements< typename TestType::RealType, typename TestType::IndexType >,
   detail::AntiDiagonalRowLengths< typename TestType::IndexType >,
   typename TestType::RealType,
   typename TestType::DeviceType,
   typename TestType::IndexType >
{
   using Real = typename TestType::RealType;
   using Device = typename TestType::DeviceType;
   using Index = typename TestType::IndexType;

   detail::AntiDiagonalRowLengths< Index > rowLengths;
   detail::AntiDiagonalMatrixElements< Real, Index > matrixElements;

   return TNL::Matrices::LambdaMatrixFactory< Real, Device, Index >::create( size, size, matrixElements, rowLengths );
}
template< typename TestType >
void
test_forElements_Range()
{
   using Real = typename TestType::RealType;
   using Index = typename TestType::IndexType;
   using Device = typename TestType::DeviceType;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< TestType >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forElements(
      matrix,
      (Index) 1,
      (Index) 4,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         // A wrong column index spoils the stored value, which is checked on the host.
         rowSumsView[ rowIdx ] = columnIdx == size - 1 - rowIdx ? value : static_cast< Real >( -1000 );
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
         // A wrong column index spoils the stored value, which is checked on the host.
         rowSumsView[ rowIdx ] = columnIdx == size - 1 - rowIdx ? value : static_cast< Real >( -1000 );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename TestType >
void
test_forAllElements()
{
   using Real = typename TestType::RealType;
   using Index = typename TestType::IndexType;
   using Device = typename TestType::DeviceType;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< TestType >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::forAllElements(
      matrix,
      [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
      {
         // A wrong column index spoils the stored value, which is checked on the host.
         rowSumsView[ rowIdx ] = columnIdx == size - 1 - rowIdx ? value : static_cast< Real >( -1000 );
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
         // A wrong column index spoils the stored value, which is checked on the host.
         rowSumsView[ rowIdx ] = columnIdx == size - 1 - rowIdx ? value : static_cast< Real >( -1000 );
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );
}

template< typename TestType >
void
test_forRows()
{
   using Real = typename TestType::RealType;
   using Index = typename TestType::IndexType;
   using Device = typename TestType::DeviceType;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using MatrixType = decltype( createAntiDiagonalMatrix< TestType >( (Index) 5 ) );
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< TestType >( size );

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

template< typename TestType >
void
test_forAllRows()
{
   using Real = typename TestType::RealType;
   using Index = typename TestType::IndexType;
   using Device = typename TestType::DeviceType;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using MatrixType = decltype( createAntiDiagonalMatrix< TestType >( (Index) 5 ) );
   using RowView = typename MatrixType::RowView;
   using ConstRowView = typename MatrixType::ConstRowView;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< TestType >( size );

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

// Tests the variants with an array of row indexes and the conditional variants. Each variant records
// the value and the column index of the only element in each processed row, which are checked on the host.
template< typename TestType >
void
test_variantsWithRowIndexesAndConditions()
{
   using Real = typename TestType::RealType;
   using Index = typename TestType::IndexType;
   using Device = typename TestType::DeviceType;
   using RealVector = TNL::Containers::Vector< Real, Device, Index >;
   using IndexVector = TNL::Containers::Vector< Index, Device, Index >;
   using MatrixType = decltype( createAntiDiagonalMatrix< TestType >( (Index) 6 ) );
   using RowView = typename MatrixType::RowView;

   const Index size = 6;
   auto matrix = createAntiDiagonalMatrix< TestType >( size );
   const auto& constMatrix = matrix;

   RealVector values( size );
   IndexVector columns( size );
   auto values_view = values.getView();
   auto columns_view = columns.getView();
   auto recordElement = [ = ] __cuda_callable__( Index rowIdx, Index localIdx, Index columnIdx, const Real& value ) mutable
   {
      values_view[ rowIdx ] = value;
      columns_view[ rowIdx ] = columnIdx;
   };
   auto recordRow = [ = ] __cuda_callable__( const RowView& row ) mutable
   {
      values_view[ row.getRowIndex() ] = row.getValue( 0 );
      columns_view[ row.getRowIndex() ] = row.getColumnIndex( 0 );
   };
   auto evenRows = [] __cuda_callable__( Index rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };
   const IndexVector rowIndexes{ 5, 0, 3, 2 };

   auto reset = [ & ]()
   {
      values.setValue( -1 );
      columns.setValue( -1 );
   };
   // Checks that exactly the given rows were processed
   auto check = [ & ]( const std::vector< Index >& processedRows )
   {
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostValues;
      TNL::Containers::Vector< Index, TNL::Devices::Host, Index > hostColumns;
      hostValues = values;
      hostColumns = columns;
      for( Index rowIdx = 0; rowIdx < size; rowIdx++ ) {
         const bool processed = std::find( processedRows.begin(), processedRows.end(), rowIdx ) != processedRows.end();
         EXPECT_EQ( hostValues[ rowIdx ], processed ? size - rowIdx : -1 ) << "row " << rowIdx;
         EXPECT_EQ( hostColumns[ rowIdx ], processed ? size - 1 - rowIdx : -1 ) << "row " << rowIdx;
      }
   };

   reset();
   TNL::Matrices::forElements( matrix, (Index) 1, (Index) 4, recordElement );
   check( { 1, 2, 3 } );

   reset();
   TNL::Matrices::forAllElements( constMatrix, recordElement );
   check( { 0, 1, 2, 3, 4, 5 } );

   reset();
   TNL::Matrices::forElements( matrix, rowIndexes, recordElement );
   check( { 5, 0, 3, 2 } );

   reset();
   TNL::Matrices::forElements( constMatrix, rowIndexes, (Index) 1, (Index) 3, recordElement );
   check( { 0, 3 } );

   reset();
   TNL::Matrices::forElementsIf( matrix, (Index) 1, (Index) 5, evenRows, recordElement );
   check( { 2, 4 } );

   reset();
   TNL::Matrices::forAllElementsIf( constMatrix, evenRows, recordElement );
   check( { 0, 2, 4 } );

   reset();
   TNL::Matrices::forElementsIf( matrix, rowIndexes, (Index) 0, (Index) 4, evenRows, recordElement );
   check( { 0, 2 } );

   reset();
   TNL::Matrices::forElementsIf( constMatrix, rowIndexes, (Index) 2, (Index) 4, evenRows, recordElement );
   check( { 2 } );

   reset();
   TNL::Matrices::forRows( matrix, rowIndexes, recordRow );
   check( { 5, 0, 3, 2 } );

   reset();
   TNL::Matrices::forRows( constMatrix, rowIndexes, (Index) 1, (Index) 3, recordRow );
   check( { 0, 3 } );

   reset();
   TNL::Matrices::forRowsIf( matrix, (Index) 1, (Index) 5, evenRows, recordRow );
   check( { 2, 4 } );

   reset();
   TNL::Matrices::forAllRowsIf( constMatrix, evenRows, recordRow );
   check( { 0, 2, 4 } );

   reset();
   TNL::Matrices::forRowsIf( matrix, rowIndexes, (Index) 0, (Index) 4, evenRows, recordRow );
   check( { 0, 2 } );

   reset();
   TNL::Matrices::forRowsIf( constMatrix, rowIndexes, (Index) 2, (Index) 4, evenRows, recordRow );
   check( { 2 } );
}

// Test fixture
template< typename TestType >
class LambdaMatrixTraverseTest : public ::testing::Test
{
protected:
   using TestType_ = TestType;
};

TYPED_TEST_SUITE_P( LambdaMatrixTraverseTest );

TYPED_TEST_P( LambdaMatrixTraverseTest, forElements_Range )
{
   test_forElements_Range< TypeParam >();
}

TYPED_TEST_P( LambdaMatrixTraverseTest, forAllElements )
{
   test_forAllElements< TypeParam >();
}

TYPED_TEST_P( LambdaMatrixTraverseTest, forRows )
{
   test_forRows< TypeParam >();
}

TYPED_TEST_P( LambdaMatrixTraverseTest, forAllRows )
{
   test_forAllRows< TypeParam >();
}

TYPED_TEST_P( LambdaMatrixTraverseTest, variantsWithRowIndexesAndConditions )
{
   test_variantsWithRowIndexesAndConditions< TypeParam >();
}

REGISTER_TYPED_TEST_SUITE_P(
   LambdaMatrixTraverseTest,
   forElements_Range,
   forAllElements,
   forRows,
   forAllRows,
   variantsWithRowIndexesAndConditions );

INSTANTIATE_TYPED_TEST_SUITE_P( LambdaMatrix, LambdaMatrixTraverseTest, LambdaMatrixTraverseTypes );

#include "../../main.h"
