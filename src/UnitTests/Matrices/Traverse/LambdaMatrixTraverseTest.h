// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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
         TNL_ASSERT_EQ( columnIdx, size - 1 - rowIdx, "wrong columnIdx for anti-diagonal matrix" );
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
         TNL_ASSERT_EQ( columnIdx, size - 1 - rowIdx, "wrong columnIdx for anti-diagonal matrix" );
         rowSumsView[ rowIdx ] = value;
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
         TNL_ASSERT_EQ( columnIdx, size - 1 - rowIdx, "wrong columnIdx for anti-diagonal matrix" );
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
         TNL_ASSERT_EQ( columnIdx, size - 1 - rowIdx, "wrong columnIdx for anti-diagonal matrix" );
         rowSumsView[ rowIdx ] = value;
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

REGISTER_TYPED_TEST_SUITE_P( LambdaMatrixTraverseTest, forElements_Range, forAllElements, forRows, forAllRows );

INSTANTIATE_TYPED_TEST_SUITE_P( LambdaMatrix, LambdaMatrixTraverseTest, LambdaMatrixTraverseTypes );

#include "../../main.h"
