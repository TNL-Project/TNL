#pragma once

#include "SparseMatrixVectorProductTest.hpp"

#include <gtest/gtest.h>

// test fixture for type-parameterized tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE_P( MatrixTest );

TYPED_TEST_P( MatrixTest, vectorProductTest_zeroMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_zeroMatrix< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_smallMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix1< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_smallMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix2< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_smallMatrix3 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix3< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_mediumSizeMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_mediumSizeMatrix1< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_mediumSizeMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_mediumSizeMatrix2< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_largeMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_largeMatrix< MatrixType >();
}

TYPED_TEST_P( MatrixTest, vectorProductTest_longRowsMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_longRowsMatrix< MatrixType >();
}

REGISTER_TYPED_TEST_SUITE_P(
   MatrixTest,
   vectorProductTest_zeroMatrix,
   vectorProductTest_smallMatrix1,
   vectorProductTest_smallMatrix2,
   vectorProductTest_smallMatrix3,
   vectorProductTest_mediumSizeMatrix1,
   vectorProductTest_mediumSizeMatrix2,
   vectorProductTest_largeMatrix,
   vectorProductTest_longRowsMatrix );
