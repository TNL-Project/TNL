#pragma once

#include "SparseDenseMatrixOperationsTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class SparseDenseMatrixOperationsTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using SymmetricMatrixType = typename Matrix::Self< RealType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
};

TYPED_TEST_SUITE( SparseDenseMatrixOperationsTest, MatrixTypes );

TYPED_TEST( SparseDenseMatrixOperationsTest, copyDenseToDenseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   copyDenseToDenseMatrix_test< MatrixType >();
}

TYPED_TEST( SparseDenseMatrixOperationsTest, copySparseToDenseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   copySparseToDenseMatrix_test< MatrixType >();
}

TYPED_TEST( SparseDenseMatrixOperationsTest, copyDenseToSparseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   copyDenseToSparseMatrix_test< MatrixType >();
}
