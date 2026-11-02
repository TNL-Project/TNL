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

TYPED_TEST( MatrixTest, sparseDenseMatrixProduct_wrongDimensions )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SparseDenseMatrixProduct_wrongDimensions< MatrixType >();
}
