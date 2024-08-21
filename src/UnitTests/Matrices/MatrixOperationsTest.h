#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

using namespace TNL;

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types< TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, short >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, short >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, long >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, long >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, short >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, short >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                      ,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, short >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, short >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, long >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, short >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, short >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                      ,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, short >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, short >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, int >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >,
                                      TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, long >,
                                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, long >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, short >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, short >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >,
                                      TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long >,
                                      TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long >
#endif
                                      >;

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, MaxNormTest1 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   // clang-format off
   DenseMatrixType dense_matrix( { { 1, 2, 3 },
                                   { 4, 5, 6 },
                                   { 7, 8, 9 } } );
   // clang-format on
   MatrixType matrix;
   matrix = dense_matrix;
   auto norm = Matrices::maxNorm( matrix );
   EXPECT_EQ( norm, 24 );
}

TYPED_TEST( MatrixTest, MaxNormTest2 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   // clang-format off
   DenseMatrixType dense_matrix( { { 1, 2, -3 },
                                   { 4, 5, -6 },
                                   { 7, 8, -9 } } );
   // clang-format on
   MatrixType matrix;
   matrix = dense_matrix;
   auto norm = Matrices::maxNorm( matrix );
   EXPECT_EQ( norm, 24 );
}

#include "../main.h"
