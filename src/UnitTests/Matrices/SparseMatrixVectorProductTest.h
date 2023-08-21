#pragma once

#include "SparseMatrixVectorProductTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename T >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = typename T::MatrixType;
   using KernelType = typename T::KernelType;
};

TYPED_TEST_SUITE( MatrixTest, MatrixAndKernelTypes );

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_smallMatrix1< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_smallMatrix2< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix3 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_smallMatrix3< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix1 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_mediumSizeMatrix1< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix2 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_mediumSizeMatrix2< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_largeMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_largeMatrix< MatrixType, KernelType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_longRowsMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;
   using KernelType = typename TestFixture::KernelType;

   test_VectorProduct_longRowsMatrix< MatrixType, KernelType >();
}
