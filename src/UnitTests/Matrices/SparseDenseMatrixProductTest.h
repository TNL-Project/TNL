#pragma once

#include "SparseDenseMatrixProductTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename T >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = typename T::MatrixType;
};

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, sparseDenseMatrixProduct_rowMajor )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProduct< MatrixType, TNL::Algorithms::Segments::RowMajorOrder >();
}

TYPED_TEST( MatrixTest, sparseDenseMatrixProduct_columnMajor )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProduct< MatrixType, TNL::Algorithms::Segments::ColumnMajorOrder >();
}

TYPED_TEST( MatrixTest, sparseDenseMatrixProduct_transposedB )
{
   using MatrixType = typename TestFixture::MatrixType;
   using TNL::Algorithms::Segments::ColumnMajorOrder;
   using TNL::Algorithms::Segments::RowMajorOrder;

   test_SparseDenseMatrixProductTransposedB< MatrixType, RowMajorOrder, RowMajorOrder >();
   test_SparseDenseMatrixProductTransposedB< MatrixType, RowMajorOrder, ColumnMajorOrder >();
   test_SparseDenseMatrixProductTransposedB< MatrixType, ColumnMajorOrder, RowMajorOrder >();
   test_SparseDenseMatrixProductTransposedB< MatrixType, ColumnMajorOrder, ColumnMajorOrder >();
}

TYPED_TEST( MatrixTest, sparseDenseMatrixProductLanesOverColumns )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProductLanesOverColumns< MatrixType, TNL::Algorithms::Segments::RowMajorOrder >();
   test_SparseDenseMatrixProductLanesOverColumns< MatrixType, TNL::Algorithms::Segments::ColumnMajorOrder >();
}

TYPED_TEST( MatrixTest, sparseDenseMatrixProductWithLambdas )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProductWithLambdas< MatrixType, TNL::Algorithms::Segments::RowMajorOrder >();
   test_SparseDenseMatrixProductWithLambdas< MatrixType, TNL::Algorithms::Segments::ColumnMajorOrder >();
}

TYPED_TEST( MatrixTest, denseSparseMatrixProduct )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_DenseSparseMatrixProduct< MatrixType, TNL::Algorithms::Segments::RowMajorOrder >();
   test_DenseSparseMatrixProduct< MatrixType, TNL::Algorithms::Segments::ColumnMajorOrder >();
}

TYPED_TEST( MatrixTest, sparseDenseMatrixProduct_wrongDimensions )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProduct_wrongDimensions< MatrixType >();
}
