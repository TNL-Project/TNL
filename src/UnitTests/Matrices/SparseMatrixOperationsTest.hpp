#pragma once

#include <functional>
#include <iostream>
#include <sstream>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

#include <gtest/gtest.h>

template< typename Matrix >
void
copyDenseToDenseMatrix_test()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   // Create source matrix
   // clang-format off
   DenseMatrixType source{
     { 1, 2, 3 },
     { 4, 5, 6 },
     { 7, 8, 9 } };
   // clang-format on

   // Create destination matrix
   DenseMatrixType destination( 3, 3 );

   // Copy matrix
   TNL::Matrices::copyDenseToDenseMatrix( destination, source );

   // Verify all elements
   EXPECT_EQ( destination.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( destination.getElement( 1, 0 ), 4.0 );
   EXPECT_EQ( destination.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination.getElement( 1, 2 ), 6.0 );
   EXPECT_EQ( destination.getElement( 2, 0 ), 7.0 );
   EXPECT_EQ( destination.getElement( 2, 1 ), 8.0 );
   EXPECT_EQ( destination.getElement( 2, 2 ), 9.0 );

   // Test with different sizes
   // clang-format off
   DenseMatrixType source2{ 
      { 10.0, 20.0, 30.0, 40.0 },
      { 50.0, 60.0, 70.0, 80.0 } };
   // clang-format on

   DenseMatrixType destination2( 2, 4 );
   TNL::Matrices::copyDenseToDenseMatrix( destination2, source2 );

   EXPECT_EQ( destination2.getElement( 0, 0 ), 10.0 );
   EXPECT_EQ( destination2.getElement( 0, 1 ), 20.0 );
   EXPECT_EQ( destination2.getElement( 0, 2 ), 30.0 );
   EXPECT_EQ( destination2.getElement( 0, 3 ), 40.0 );
   EXPECT_EQ( destination2.getElement( 1, 0 ), 50.0 );
   EXPECT_EQ( destination2.getElement( 1, 1 ), 60.0 );
   EXPECT_EQ( destination2.getElement( 1, 2 ), 70.0 );
   EXPECT_EQ( destination2.getElement( 1, 3 ), 80.0 );
}

template< typename Matrix >
void
copySparseToDenseMatrix_test()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   // Create sparse source matrix
   // clang-format off
  // /  1  2  0 |
  // |  0  5  6 |
  // \  7  0  9 /
  Matrix source( 3, 3,
  { { 0, 0, 1 }, { 0, 1, 2 },
                 { 1, 1, 5 }, { 1, 2, 6 },
    { 2, 0, 7 },                           { 2, 2, 9 } } );
   // clang-format on

   // Create dense destination matrix
   DenseMatrixType destination( 3, 3 );

   // Copy matrix
   TNL::Matrices::copySparseToDenseMatrix( destination, source );

   // Verify all elements
   EXPECT_EQ( destination.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination.getElement( 0, 2 ), 0.0 );
   EXPECT_EQ( destination.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination.getElement( 1, 2 ), 6.0 );
   EXPECT_EQ( destination.getElement( 2, 0 ), 7.0 );
   EXPECT_EQ( destination.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( destination.getElement( 2, 2 ), 9.0 );

   // Test with different sizes (non-square matrix)
   // clang-format off
  // / 10  0 20  0 |
  // \  0 50  0 70 /
  Matrix source2( 2, 4,
  { { 0, 0, 10 },               { 0, 2, 20 },
                  { 1, 1, 50 },               { 1, 3, 70 } } );
   // clang-format on

   DenseMatrixType destination2( 2, 4 );
   TNL::Matrices::copySparseToDenseMatrix( destination2, source2 );

   EXPECT_EQ( destination2.getElement( 0, 0 ), 10.0 );
   EXPECT_EQ( destination2.getElement( 0, 1 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 0, 2 ), 20.0 );
   EXPECT_EQ( destination2.getElement( 0, 3 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 1 ), 50.0 );
   EXPECT_EQ( destination2.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 3 ), 70.0 );

   // Test with empty sparse matrix
   Matrix source3( 2, 2 );
   DenseMatrixType destination3( 2, 2 );
   TNL::Matrices::copySparseToDenseMatrix( destination3, source3 );

   EXPECT_EQ( destination3.getElement( 0, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 0, 1 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 1 ), 0.0 );
}

template< typename Matrix >
void
compressSparseMatrix_test()
{
   // clang-format off
   // /  1  2  1  1  4 |
   // |  1  6  1  1  9 |
   // |  1  1 12 13  1 |
   // | 15 16  1  1  1 |
   // \  1  1  1  1 24 /
   Matrix A( 5, 5,
   { { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 1 }, { 0, 3, 1 }, { 0, 4, 2 },
     { 1, 0, 1 }, { 1, 1, 2 }, { 1, 2, 1 }, { 1, 3, 1 }, { 1, 4, 2 },
     { 2, 0, 1 }, { 2, 1, 1 }, { 2, 2, 2 }, { 2, 3, 2 }, { 2, 4, 1 },
     { 3, 0, 2 }, { 3, 1, 2 }, { 3, 2, 1 }, { 3, 3, 1 }, { 3, 4, 1 },
     { 4, 0, 1 }, { 4, 1, 1 }, { 4, 2, 1 }, { 4, 3, 1 }, { 4, 4, 2 } } );
   // clang-format on

   EXPECT_EQ( A.getValues().getSize(), 25 );
   A.getValues() -= 1.0;
   compressSparseMatrix( A );
   EXPECT_EQ( A.getValues().getSize(), 9 );

   // 1-st row
   EXPECT_EQ( A.getElement( 0, 0 ), 0.0 );
   EXPECT_EQ( A.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( A.getElement( 0, 2 ), 0.0 );
   EXPECT_EQ( A.getElement( 0, 3 ), 0.0 );
   EXPECT_EQ( A.getElement( 0, 4 ), 1.0 );

   // 2-nd row
   EXPECT_EQ( A.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( A.getElement( 1, 1 ), 1.0 );
   EXPECT_EQ( A.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( A.getElement( 1, 3 ), 0.0 );
   EXPECT_EQ( A.getElement( 1, 4 ), 1.0 );

   // 3-rd row
   EXPECT_EQ( A.getElement( 2, 0 ), 0.0 );
   EXPECT_EQ( A.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( A.getElement( 2, 2 ), 1.0 );
   EXPECT_EQ( A.getElement( 2, 3 ), 1.0 );
   EXPECT_EQ( A.getElement( 2, 4 ), 0.0 );

   // 4-th row
   EXPECT_EQ( A.getElement( 3, 0 ), 1.0 );
   EXPECT_EQ( A.getElement( 3, 1 ), 1.0 );
   EXPECT_EQ( A.getElement( 3, 2 ), 0.0 );
   EXPECT_EQ( A.getElement( 3, 3 ), 0.0 );
   EXPECT_EQ( A.getElement( 3, 4 ), 0.0 );

   // 5-th row
   EXPECT_EQ( A.getElement( 4, 0 ), 0.0 );
   EXPECT_EQ( A.getElement( 4, 1 ), 0.0 );
   EXPECT_EQ( A.getElement( 4, 2 ), 0.0 );
   EXPECT_EQ( A.getElement( 4, 3 ), 0.0 );
   EXPECT_EQ( A.getElement( 4, 4 ), 1.0 );
}
