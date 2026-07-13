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

/**
 * Creates a 5x5 upper-triangular matrix with zero diagonal:
 *
 *    /  0  1  2  3  4 \
 *    |  0  0 12 13 14 |
 *    |  0  0  0  0  0 |   <- empty row (no elements)
 *    |  0  0  0  0 34 |
 *    \  0  0  0  0  0 /   <- empty row (no elements)
 *
 * Element values are computed as rowIdx * 10 + columnIdx.
 * Rows 2 and 4 are empty, which tests the empty-row behavior.
 */
template< typename Real, typename Index >
auto
createUpperTriangularMatrixWithEmptyRows( Index size )
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;

   auto rowLengths = [ = ] __cuda_callable__( Index rows, Index columns, Index rowIdx ) -> Index
   {
      // Row has elements only for columns strictly above the diagonal
      if( rowIdx >= columns - 1 )
         return 0;
      // Rows 2 and 4 are artificially made empty
      if( rowIdx == 2 )
         return 0;
      return columns - 1 - rowIdx;
   };

   auto matrixElements =
      [ = ] __cuda_callable__( Index rows, Index columns, Index rowIdx, Index localIdx, Index & columnIdx, Real & value )
   {
      columnIdx = rowIdx + 1 + localIdx;
      value = (Real) ( rowIdx * 10 + columnIdx );
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

template< typename Real, typename Index >
void
test_reduceRowsWithArgument()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using IndexVectorType = TNL::Containers::Vector< Index, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType maxValues( size, 0 );
   IndexVectorType maxColumns( size, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      return value;
   };
   auto reduce = [] __cuda_callable__( Real & a, const Real& b, Index& aIdx, Index bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__( Index row, Index localIdx, Index columnIdx, const Real& value, bool emptyRow ) mutable
   {
      maxValuesView[ row ] = value;
      if( ! emptyRow )
         maxColumnsView[ row ] = columnIdx;
   };

   // reduceAllRowsWithArgument — each row has one element on the anti-diagonal
   // Row 0 -> col 4 value 5, Row 1 -> col 3 value 4, etc.
   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, store, (Real) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 5 );
   EXPECT_EQ( maxColumns.getElement( 0 ), 4 );
   EXPECT_EQ( maxValues.getElement( 1 ), 4 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 3 );
   EXPECT_EQ( maxValues.getElement( 2 ), 3 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 2 );
   EXPECT_EQ( maxValues.getElement( 3 ), 2 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 1 );
   EXPECT_EQ( maxValues.getElement( 4 ), 1 );
   EXPECT_EQ( maxColumns.getElement( 4 ), 0 );

   // reduceRowsWithArgument (range)
   const auto constMatrix = matrix;
   maxValues = 0;
   maxColumns = -1;
   TNL::Matrices::reduceRowsWithArgument( constMatrix, (Index) 1, (Index) 4, fetch, reduce, store, (Real) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by range
   EXPECT_EQ( maxValues.getElement( 1 ), 4 );
   EXPECT_EQ( maxColumns.getElement( 1 ), 3 );
   EXPECT_EQ( maxValues.getElement( 2 ), 3 );
   EXPECT_EQ( maxColumns.getElement( 2 ), 2 );
   EXPECT_EQ( maxValues.getElement( 3 ), 2 );
   EXPECT_EQ( maxColumns.getElement( 3 ), 1 );
   EXPECT_EQ( maxValues.getElement( 4 ), 0 );  // skipped by range
}

template< typename Real, typename Index >
void
test_reduceRowsIf()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using IndexVectorType = TNL::Containers::Vector< Index, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType rowSums( size, 0 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      return value;
   };
   auto condition = [] __cuda_callable__( Index rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;  // Process only even rows: 0, 2, 4
   };
   auto store = [ = ] __cuda_callable__( Index rank, Index rowIdx, const Real& value ) mutable
   {
      rowSumsView[ rowIdx ] = value;
   };

   // reduceAllRowsIf (range-based condition)
   TNL::Matrices::reduceAllRowsIf( matrix, condition, fetch, TNL::Plus{}, store, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );  // processed
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );  // processed
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );  // processed

   // reduceRowsIf (range)
   const auto constMatrix = matrix;
   rowSums = 0;
   TNL::Matrices::reduceRowsIf( constMatrix, (Index) 1, (Index) 5, condition, fetch, TNL::Plus{}, store, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );  // skipped by range
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // skipped by condition
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );  // processed
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // skipped by condition
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );  // processed

   // reduceRowsIf (array) — condition receives the row index, not the position
   rowSums = 0;
   IndexVectorType rowIndexes{ 0, 1, 2, 4 };
   TNL::Matrices::reduceRowsIf(
      constMatrix, rowIndexes, (Index) 0, rowIndexes.getSize(), condition, fetch, TNL::Plus{}, store, (Real) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 5 );  // processed (rowIdx=0, even)
   EXPECT_EQ( rowSums.getElement( 1 ), 0 );  // skipped by condition (rowIdx=1, odd)
   EXPECT_EQ( rowSums.getElement( 2 ), 3 );  // processed (rowIdx=2, even)
   EXPECT_EQ( rowSums.getElement( 3 ), 0 );  // not in array
   EXPECT_EQ( rowSums.getElement( 4 ), 1 );  // processed (rowIdx=4, even)
}

template< typename Real, typename Index >
void
test_reduceRowsWithArgumentIf()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using IndexVectorType = TNL::Containers::Vector< Index, Device, Index >;

   const Index size = 5;
   auto matrix = createAntiDiagonalMatrix< Real, Index >( size );

   VectorType maxValues( size, 0 );
   IndexVectorType maxColumns( size, -1 );
   auto maxValuesView = maxValues.getView();
   auto maxColumnsView = maxColumns.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      return value;
   };
   auto condition = [] __cuda_callable__( Index rowIdx ) -> bool
   {
      return rowIdx >= 2;  // Process only rows with index >= 2
   };
   auto reduce = [] __cuda_callable__( Real & a, const Real& b, Index& aIdx, Index bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__(
                   Index rank, Index row, Index localIdx, Index columnIdx, const Real& value, bool emptyRow ) mutable
   {
      maxValuesView[ row ] = value;
      if( ! emptyRow )
         maxColumnsView[ row ] = columnIdx;
   };

   // reduceAllRowsWithArgumentIf
   TNL::Matrices::reduceAllRowsWithArgumentIf( matrix, condition, fetch, reduce, store, (Real) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition
   EXPECT_EQ( maxColumns.getElement( 0 ), -1 );
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // skipped by condition
   EXPECT_EQ( maxColumns.getElement( 1 ), -1 );
   EXPECT_EQ( maxValues.getElement( 2 ), 3 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 2 );
   EXPECT_EQ( maxValues.getElement( 3 ), 2 );  // processed
   EXPECT_EQ( maxColumns.getElement( 3 ), 1 );
   EXPECT_EQ( maxValues.getElement( 4 ), 1 );  // processed
   EXPECT_EQ( maxColumns.getElement( 4 ), 0 );

   // reduceRowsWithArgumentIf (array) — condition receives the row index
   const auto constMatrix = matrix;
   maxValues = 0;
   maxColumns = -1;
   IndexVectorType rowIndexes{ 0, 2, 4 };
   TNL::Matrices::reduceRowsWithArgumentIf(
      constMatrix, rowIndexes, (Index) 0, rowIndexes.getSize(), condition, fetch, reduce, store, (Real) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 0 );  // skipped by condition (rowIdx=0)
   EXPECT_EQ( maxValues.getElement( 1 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 2 ), 3 );  // processed
   EXPECT_EQ( maxColumns.getElement( 2 ), 2 );
   EXPECT_EQ( maxValues.getElement( 3 ), 0 );  // not in array
   EXPECT_EQ( maxValues.getElement( 4 ), 1 );  // processed
   EXPECT_EQ( maxColumns.getElement( 4 ), 0 );
}

template< typename Real, typename Index >
void
test_emptyRows()
{
   using Device = LAMBDA_MATRIX_REDUCE_TEST_DEVICE;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   const Index size = 5;
   auto matrix = createUpperTriangularMatrixWithEmptyRows< Real, Index >( size );

   VectorType rowSums( size, -1 );
   auto rowSumsView = rowSums.getView();

   auto fetch = [ = ] __cuda_callable__( Index row, Index columnIdx, const Real& value ) -> Real
   {
      return value;
   };
   auto keep = [ = ] __cuda_callable__( Index row, const Real& value ) mutable
   {
      rowSumsView[ row ] = value;
   };

   // Basic reduction: empty rows should get the identity value (0)
   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, keep, (Real) 0 );

   // Row 0: cols 1,2,3,4 -> values 1,2,3,4 -> sum = 10
   EXPECT_EQ( rowSums.getElement( 0 ), 10 );
   // Row 1: cols 2,3,4 -> values 12,13,14 -> sum = 39
   EXPECT_EQ( rowSums.getElement( 1 ), 39 );
   // Row 2: empty -> identity = 0
   EXPECT_EQ( rowSums.getElement( 2 ), 0 );
   // Row 3: col 4 -> value 34 -> sum = 34
   EXPECT_EQ( rowSums.getElement( 3 ), 34 );
   // Row 4: empty -> identity = 0
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );

   // WithArgument reduction: empty rows should get emptyRow = true
   VectorType maxValues( size, -1 );
   auto maxValuesView = maxValues.getView();
   auto reduce = [] __cuda_callable__( Real & a, const Real& b, Index& aIdx, Index bIdx )
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
   };
   auto store = [ = ] __cuda_callable__( Index row, Index localIdx, Index columnIdx, const Real& value, bool emptyRow ) mutable
   {
      if( emptyRow )
         maxValuesView[ row ] = -1;  // signal that emptyRow was true
      else
         maxValuesView[ row ] = value;
   };

   TNL::Matrices::reduceAllRowsWithArgument( matrix, fetch, reduce, store, (Real) 0 );

   EXPECT_EQ( maxValues.getElement( 0 ), 4 );  // max of {1,2,3,4}
   EXPECT_EQ( maxValues.getElement( 1 ), 14 );  // max of {12,13,14}
   EXPECT_EQ( maxValues.getElement( 2 ), -1 );  // empty row
   EXPECT_EQ( maxValues.getElement( 3 ), 34 );  // max of {34}
   EXPECT_EQ( maxValues.getElement( 4 ), -1 );  // empty row
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

TEST( LambdaMatrixReduceTest, reduceRowsWithArgument )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceRowsWithArgument< double, int >();
   test_reduceRowsWithArgument< float, long >();
}

TEST( LambdaMatrixReduceTest, reduceRowsIf )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceRowsIf< double, int >();
   test_reduceRowsIf< float, long >();
}

TEST( LambdaMatrixReduceTest, reduceRowsWithArgumentIf )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_reduceRowsWithArgumentIf< double, int >();
   test_reduceRowsWithArgumentIf< float, long >();
}

TEST( LambdaMatrixReduceTest, emptyRows )
{
   using namespace LambdaMatrixReduceTestNamespace;
   test_emptyRows< double, int >();
   test_emptyRows< float, long >();
}

#undef LAMBDA_MATRIX_REDUCE_TEST_DEVICE
