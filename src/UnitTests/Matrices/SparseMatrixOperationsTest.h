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
      copySparseToSparseMatrixWithDifferentDevice_test<
         HostMatrixType,
         MatrixType,
         HostSymmetricMatrixType,
         SymmetricMatrixType >();
      copySparseToSparseMatrixWithDifferentDevice_test<
         MatrixType,
         HostMatrixType,
         SymmetricMatrixType,
         HostSymmetricMatrixType >();
   }
}

TYPED_TEST( SparseMatrixOperationsTest, copySparseMatrixElementsAcrossDevicesInChunks )
{
   using MatrixType = typename TestFixture::MatrixType;
   using SymmetricMatrixType = typename TestFixture::SymmetricMatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;

   // In host builds, Sequential serves as the other device, so the copy across devices is tested as well
   using OtherDeviceType =
      std::conditional_t< std::is_same_v< DeviceType, TNL::Devices::Host >, TNL::Devices::Sequential, TNL::Devices::Host >;
   using OtherMatrixType = typename MatrixType::template Self< RealType, OtherDeviceType >;
   using OtherSymmetricMatrixType = typename SymmetricMatrixType::template Self< RealType, OtherDeviceType >;

   copySparseMatrixElementsAcrossDevicesInChunks_test< OtherMatrixType, MatrixType, OtherSymmetricMatrixType >();
   copySparseMatrixElementsAcrossDevicesInChunks_test< MatrixType, OtherMatrixType, SymmetricMatrixType >();
}

TYPED_TEST( SparseMatrixOperationsTest, assignmentOperator )
{
   using MatrixType = typename TestFixture::MatrixType;
   using SymmetricMatrixType = typename TestFixture::SymmetricMatrixType;

   if constexpr( ! std::is_same_v< typename MatrixType::RealType, long > ) {
      assignmentOperator_test< MatrixType, SymmetricMatrixType >();
   }
}

TYPED_TEST( SparseMatrixOperationsTest, assignmentOperatorWithDifferentDevice )
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
      if constexpr( ! std::is_same_v< typename MatrixType::RealType, long > ) {
         assignmentOperatorWithDifferentDevice_test<
            HostMatrixType,
            MatrixType,
            HostSymmetricMatrixType,
            SymmetricMatrixType >();
         assignmentOperatorWithDifferentDevice_test<
            MatrixType,
            HostMatrixType,
            SymmetricMatrixType,
            HostSymmetricMatrixType >();
      }
   }
}

TYPED_TEST( SparseMatrixOperationsTest, compressSparseMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   compressSparseMatrix_test< MatrixType >();
}
