#pragma once

#include "SparseMatrixVectorProductTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename T >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = typename T::MatrixType;
};

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, vectorProductTest_zeroMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_zeroMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix1< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix2< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix3 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_smallMatrix3< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_mediumSizeMatrix1< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_mediumSizeMatrix2< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_largeMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_largeMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_longRowsMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct_longRowsMatrix< MatrixType >();
}
