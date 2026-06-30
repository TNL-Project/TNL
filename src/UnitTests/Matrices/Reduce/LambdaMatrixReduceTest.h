// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <gtest/gtest.h>

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   #define LAMBDA_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Host
#elif defined( __CUDACC__ )
   #define LAMBDA_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Cuda
#elif defined( __HIP__ )
   #define LAMBDA_MATRIX_REDUCE_TEST_DEVICE TNL::Devices::Hip
#endif

namespace LambdaMatrixReduceTestNamespace {

template< typename Real, typename Index >
auto
createAntiDiagonalMatrix( Index size )
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;

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
test_reduceRows()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      EXPECT_EQ( columnIdx, size - 1 - row );
      return value;
   };
   auto reduce = [] __cuda_callable__( Real & sum, const Real& value ) -> Real
   {
      return sum + value;
   };
   auto keep = [ = ] __cuda_callable__( Index row, const Real& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   TNL::Matrices::reduceRows( matrix, (Index) 1, (Index) 4, fetch, reduce, keep, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::reduceRows( constMatrix, (Index) 1, (Index) 4, fetch, reduce, keep, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
}

template< typename Real, typename Index >
void
test_reduceAllRows_explicit_identity()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      EXPECT_EQ( columnIdx, size - 1 - row );
      return value;
   };
   auto reduce = [] __cuda_callable__( Real & sum, const Real& value ) -> Real
   {
      return sum + value;
   };
   auto keep = [ = ] __cuda_callable__( Index row, const Real& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   TNL::Matrices::reduceAllRows( matrix, fetch, reduce, keep, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::reduceAllRows( constMatrix, fetch, reduce, keep, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );
}

template< typename Real, typename Index >
void
test_reduceAllRows_deduced_identity()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      EXPECT_EQ( columnIdx, size - 1 - row );
      return value;
   };
   auto keep = [ = ] __cuda_callable__( Index row, const Real& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, keep );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );

   const auto constMatrix = matrix;
   rowSums = 0;

   TNL::Matrices::reduceAllRows( constMatrix, fetch, TNL::Plus{}, keep );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );
   EXPECT_EQ( rowSums.getElement( 1 ), 4 );
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );
   EXPECT_EQ( rowSums.getElement( 3 ), 2 );
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );
}

}  // namespace LambdaMatrixReduceTestNamespace

TEST( LambdaMatrixReduceTest, reduceRows )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceRows< double, int >();
   test_reduceRows< float, long >();
}

TEST( LambdaMatrixReduceTest, reduceAllRows_explicit_identity )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceAllRows_explicit_identity< double, int >();
   test_reduceAllRows_explicit_identity< float, long >();
}

TEST( LambdaMatrixReduceTest, reduceAllRows_deduced_identity )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceAllRows_deduced_identity< double, int >();
   test_reduceAllRows_deduced_identity< float, long >();
}

#undef LAMBDA_MATRIX_REDUCE_TEST_DEVICE
