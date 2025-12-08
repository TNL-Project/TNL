#pragma once

#include "SparseMatrixOperationsTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class SparseMatrixOperationsTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE( SparseMatrixOperationsTest, MatrixTypes );

TYPED_TEST( SparseMatrixOperationsTest, copyDenseToDenseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   copyDenseToDenseMatrix_test< MatrixType >();
}

TYPED_TEST( SparseMatrixOperationsTest, compressSparseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   compressSparseMatrix_test< MatrixType >();
}
