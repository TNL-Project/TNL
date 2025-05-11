#include <functional>
#include <iostream>
#include <sstream>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/CusparseCSRMatrix.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>

#include <gtest/gtest.h>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename RealType >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda >;
};

// types for which types MatrixTest is instantiated
using RealTypes = ::testing::Types< float, double >;

TYPED_TEST_SUITE( MatrixTest, RealTypes );

template< typename Matrix >
void
test_VectorProduct_smallMatrix1()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using CusparseCSRMatrix = TNL::Matrices::CusparseCSRMatrix< Matrix >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  0  0  0 \
    *    |  0  2  0  3 |
    *    |  0  4  0  0 |
    *    \  0  0  5  0 /
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1;
   m_1.reset();
   m_1.setDimensions( m_rows_1, m_cols_1 );
   typename Matrix::RowCapacitiesType rowLengths_1{ 1, 2, 1, 1 };
   m_1.setRowCapacities( rowLengths_1 );

   IndexType value_1 = 1;
   m_1.setElement( 0, 0, value_1++ );  // 0th row

   m_1.setElement( 1, 1, value_1++ );  // 1st row
   m_1.setElement( 1, 3, value_1++ );

   m_1.setElement( 2, 1, value_1++ );  // 2nd row

   m_1.setElement( 3, 2, value_1++ );  // 3rd row

   VectorType inVector_1( m_cols_1, 2 );
   VectorType inVector_2( m_cols_1, 1 );
   VectorType outVector_1( m_rows_1, 0 );
   VectorType outVector_2( m_rows_1, 0 );

   CusparseCSRMatrix cusparse_m1;
   cusparse_m1.init( m_1 );
   cusparse_m1.vectorProduct( inVector_1, outVector_1 );

   EXPECT_EQ( outVector_1.getElement( 0 ), RealType{ 2 } );
   EXPECT_EQ( outVector_1.getElement( 1 ), RealType{ 10 } );
   EXPECT_EQ( outVector_1.getElement( 2 ), RealType{ 8 } );
   EXPECT_EQ( outVector_1.getElement( 3 ), RealType{ 10 } );

   outVector_1 = 0;
   cusparse_m1.vectorsProduct( inVector_1, inVector_2, outVector_1, outVector_2 );

   EXPECT_EQ( outVector_1.getElement( 0 ), RealType{ 2 } );
   EXPECT_EQ( outVector_1.getElement( 1 ), RealType{ 10 } );
   EXPECT_EQ( outVector_1.getElement( 2 ), RealType{ 8 } );
   EXPECT_EQ( outVector_1.getElement( 3 ), RealType{ 10 } );

   EXPECT_EQ( outVector_2.getElement( 0 ), RealType{ 1 } );
   EXPECT_EQ( outVector_2.getElement( 1 ), RealType{ 5 } );
   EXPECT_EQ( outVector_2.getElement( 2 ), RealType{ 4 } );
   EXPECT_EQ( outVector_2.getElement( 3 ), RealType{ 5 } );
}

template< typename Matrix >
void
test_VectorProduct_smallMatrix2()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using CusparseCSRMatrix = TNL::Matrices::CusparseCSRMatrix< Matrix >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  0  0  4 |
    *    |  5  6  7  0 |
    *    \  0  8  0  0 /
    */

   const IndexType m_rows_2 = 4;
   const IndexType m_cols_2 = 4;

   Matrix m_2( m_rows_2, m_cols_2 );
   typename Matrix::RowCapacitiesType rowLengths_2{ 3, 1, 3, 1 };
   m_2.setRowCapacities( rowLengths_2 );

   IndexType value_2 = 1;
   for( IndexType i = 0; i < 3; i++ )  // 0th row
      m_2.setElement( 0, i, value_2++ );

   m_2.setElement( 1, 3, value_2++ );  // 1st row

   for( IndexType i = 0; i < 3; i++ )  // 2nd row
      m_2.setElement( 2, i, value_2++ );

   for( IndexType i = 1; i < 2; i++ )  // 3rd row
      m_2.setElement( 3, i, value_2++ );

   VectorType inVector_2;
   inVector_2.setSize( m_cols_2 );
   for( IndexType i = 0; i < inVector_2.getSize(); i++ )
      inVector_2.setElement( i, 2 );

   VectorType outVector_2;
   outVector_2.setSize( m_rows_2 );
   for( IndexType j = 0; j < outVector_2.getSize(); j++ )
      outVector_2.setElement( j, 0 );

   CusparseCSRMatrix cusparse_m2;
   cusparse_m2.init( m_2 );
   cusparse_m2.vectorProduct( inVector_2, outVector_2 );

   EXPECT_EQ( outVector_2.getElement( 0 ), RealType{ 12 } );
   EXPECT_EQ( outVector_2.getElement( 1 ), RealType{ 8 } );
   EXPECT_EQ( outVector_2.getElement( 2 ), RealType{ 36 } );
   EXPECT_EQ( outVector_2.getElement( 3 ), RealType{ 16 } );

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_2_transposed;
      m_2_transposed.getTransposition( m_2 );
      VectorType inVector_1_transposed( m_rows_2, 1.0 );
      VectorType outVector_1_transposed( m_cols_2, 0.0 );
      VectorType outVector_2_transposed( m_cols_2, 0.0 );
      m_2_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed );
      m_2.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
      EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
   }
}

template< typename Matrix >
void
test_VectorProduct_smallMatrix3()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using CusparseCSRMatrix = TNL::Matrices::CusparseCSRMatrix< Matrix >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  4  5  6 |
    *    |  7  8  9  0 |
    *    \  0 10 11 12 /
    */

   const IndexType m_rows_3 = 4;
   const IndexType m_cols_3 = 4;

   Matrix m_3( m_rows_3, m_cols_3 );
   typename Matrix::RowCapacitiesType rowLengths_3{ 3, 3, 3, 3 };
   m_3.setRowCapacities( rowLengths_3 );

   IndexType value_3 = 1;
   for( IndexType i = 0; i < 3; i++ )  // 0th row
      m_3.setElement( 0, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )
      m_3.setElement( 1, i, value_3++ );  // 1st row

   for( IndexType i = 0; i < 3; i++ )  // 2nd row
      m_3.setElement( 2, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )  // 3rd row
      m_3.setElement( 3, i, value_3++ );

   VectorType inVector_3;
   inVector_3.setSize( m_cols_3 );
   for( IndexType i = 0; i < inVector_3.getSize(); i++ )
      inVector_3.setElement( i, 2 );

   VectorType outVector_3;
   outVector_3.setSize( m_rows_3 );
   for( IndexType j = 0; j < outVector_3.getSize(); j++ )
      outVector_3.setElement( j, 0 );

   CusparseCSRMatrix cusparse_m3;
   cusparse_m3.init( m_3 );
   cusparse_m3.vectorProduct( inVector_3, outVector_3 );

   EXPECT_EQ( outVector_3.getElement( 0 ), RealType{ 12 } );
   EXPECT_EQ( outVector_3.getElement( 1 ), RealType{ 30 } );
   EXPECT_EQ( outVector_3.getElement( 2 ), RealType{ 48 } );
   EXPECT_EQ( outVector_3.getElement( 3 ), RealType{ 66 } );

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_3_transposed;
      m_3_transposed.getTransposition( m_3 );
      VectorType inVector_1_transposed( m_rows_3, 1.0 );
      VectorType outVector_1_transposed( m_cols_3, 0.0 );
      VectorType outVector_2_transposed( m_cols_3, 0.0 );
      m_3_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed );
      m_3.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
      EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
   }
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

#include "../main.h"
