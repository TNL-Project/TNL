#pragma once

#include "SparseMatrixOperationsTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class SparseMatrixOperationsTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using SymmetricMatrixType =
      typename Matrix::template Self< RealType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
};

TYPED_TEST_SUITE( SparseMatrixOperationsTest, MatrixTypes );

TYPED_TEST( SparseMatrixOperationsTest, copySparseToSparseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;
   using SymmetricMatrixType = typename TestFixture::SymmetricMatrixType;

   copySparseToSparseMatrix_test< MatrixType, SymmetricMatrixType >();
}

TYPED_TEST( SparseMatrixOperationsTest, copySparseToSparseMatrixWithDifferentDevice )
{
   using MatrixType = typename TestFixture::MatrixType;
   using SymmetricMatrixType = typename TestFixture::SymmetricMatrixType;
   using DeviceType = typename MatrixType::DeviceType;

   using HostMatrixType = typename MatrixType::template Self< typename MatrixType::RealType, TNL::Devices::Host >;
   using HostSymmetricMatrixType =
      typename SymmetricMatrixType::template Self< typename SymmetricMatrixType::RealType, TNL::Devices::Host >;

   if constexpr( ! std::is_same_v< DeviceType, TNL::Devices::Host >
                 && ! std::is_same_v< DeviceType, TNL::Devices::Sequential > )
   {
      copySparseToSparseMatrixWithDifferentDevice_test< HostMatrixType,
                                                        MatrixType,
                                                        HostSymmetricMatrixType,
                                                        SymmetricMatrixType >();
      copySparseToSparseMatrixWithDifferentDevice_test< MatrixType,
                                                        HostMatrixType,
                                                        SymmetricMatrixType,
                                                        HostSymmetricMatrixType >();
   }
}

TYPED_TEST( SparseMatrixOperationsTest, compressSparseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   compressSparseMatrix_test< MatrixType >();
}
