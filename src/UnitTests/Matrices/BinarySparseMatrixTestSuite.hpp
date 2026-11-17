#pragma once

#include <algorithm>
#include <string>

#include "BinarySparseMatrixTest.hpp"

#include <gtest/gtest.h>

// test fixture for type-parameterized tests of binary sparse matrices
template< typename Matrix >
class BinaryMatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE_P( BinaryMatrixTest );

TYPED_TEST_P( BinaryMatrixTest, setDimensionsTest )
{
   test_SetDimensions< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, setRowCapacitiesTest )
{
   test_SetRowCapacities< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, setLikeTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, resetTest )
{
   test_Reset< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, getRowTest )
{
   test_GetRow< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, setElementTest )
{
   test_SetElement< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, vectorProductTest )
{
   test_VectorProduct< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, reduceRows )
{
   test_reduceRows< typename TestFixture::MatrixType >();
}

TYPED_TEST_P( BinaryMatrixTest, saveAndLoadTest )
{
   // the file name is unique for each instantiation, because the tests of different formats may run in parallel
   std::string fileName = std::string( "test_" ) + ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
   std::replace( fileName.begin(), fileName.end(), '/', '_' );
   test_SaveAndLoad< typename TestFixture::MatrixType >( fileName.c_str() );
}

TYPED_TEST_P( BinaryMatrixTest, comparisonTest )
{
   test_Comparison< typename TestFixture::MatrixType >();
}

REGISTER_TYPED_TEST_SUITE_P(
   BinaryMatrixTest,
   setDimensionsTest,
   setRowCapacitiesTest,
   setLikeTest,
   resetTest,
   getRowTest,
   setElementTest,
   vectorProductTest,
   reduceRows,
   saveAndLoadTest,
   comparisonTest );
