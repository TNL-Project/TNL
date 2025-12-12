#pragma once

#include <tuple>
#include "MatrixOperationsTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Triple >
class MatrixOperationsTest : public ::testing::Test
{
protected:
   using TripleType = Triple;
};

using MatrixTypes = ::testing::Types< std::tuple< double, TNL::Devices::Sequential, int >,
                                      std::tuple< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                      ,
                                      std::tuple< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                      ,
                                      std::tuple< double, TNL::Devices::Hip, int >
#endif
                                      >;

TYPED_TEST_SUITE( MatrixOperationsTest, MatrixTypes );

TYPED_TEST( MatrixOperationsTest, getDiagonal_DenseMatrix )
{
   using TripleType = typename TestFixture::TripleType;

   getDiagonal_DenseMatrix_test< TripleType >();
}

TYPED_TEST( MatrixOperationsTest, getDiagonal_SparseMatrix )
{
   using TripleType = typename TestFixture::TripleType;

   getDiagonal_SparseMatrix_test< TripleType >();
}
