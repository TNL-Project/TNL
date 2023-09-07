#include <iostream>
#include <functional>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Math.h>
#include <TNL/Arithmetics/Complex.h>

using Dense_host_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >;
using Dense_host_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int >;

using Dense_cuda_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >;
using Dense_cuda_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int >;

static const char* TEST_FILE_NAME = "test_DenseMatrixTest.tnl";

#include <type_traits>

#include <gtest/gtest.h>

void
test_GetSerializationType()
{
   using namespace TNL::Algorithms::Segments;
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< float, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< int, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< float, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< int, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< float, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< int, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< float, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ),
              TNL::String( "Matrices::DenseMatrix< int, [any_device], int, ColumnMajorOrder >" ) );
}

template< typename Matrix >
void
test_SetDimensions()
{
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 9;
   const IndexType cols = 8;

   Matrix m;
   m.setDimensions( rows, cols );

   EXPECT_EQ( m.getRows(), 9 );
   EXPECT_EQ( m.getColumns(), 8 );
}

template< typename Matrix1, typename Matrix2 >
void
test_SetLike()
{
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

   Matrix1 m1;
   m1.reset();
   m1.setDimensions( rows + 1, cols + 2 );

   Matrix2 m2;
   m2.reset();
   m2.setDimensions( rows, cols );

   m1.setLike( m2 );

   EXPECT_EQ( m1.getRows(), m2.getRows() );
   EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void
test_SetElements()
{
   using RealType = typename Matrix::RealType;

   Matrix m( {
      { 1, 2, 3 },
      { 4, 5, 6 },
      { 7, 8, 9 },
   } );

   EXPECT_EQ( m.getRows(), 3 );
   EXPECT_EQ( m.getColumns(), 3 );
   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 4 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 5 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 7 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 8 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 9 } );
}

template< typename Matrix >
void
test_GetCompressedRowLengths()
{
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 10;
   const IndexType cols = 11;

   Matrix m( rows, cols );

   // Insert values into the rows.
   IndexType value = 1;

   for( IndexType i = 0; i < 3; i++ )  // 0th row
      m.setElement( 0, i, value++ );

   for( IndexType i = 0; i < 3; i++ )  // 1st row
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 1; i++ )  // 2nd row
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )  // 3rd row
      m.setElement( 3, i, value++ );

   for( IndexType i = 0; i < 3; i++ )  // 4th row
      m.setElement( 4, i, value++ );

   for( IndexType i = 0; i < 4; i++ )  // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 5; i++ )  // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 6; i++ )  // 7th row
      m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )  // 8th row
      m.setElement( 8, i, value++ );

   for( IndexType i = 0; i < 8; i++ )  // 9th row
      m.setElement( 9, i, value++ );

   typename Matrix::RowCapacitiesType rowLengths;
   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::RowCapacitiesType correctRowLengths{ 3, 3, 1, 2, 3, 4, 5, 6, 7, 8 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix >
void
test_GetAllocatedElementsCount()
{
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   EXPECT_EQ( m.getAllocatedElementsCount(), 42 );
}

template< typename Matrix >
void
test_GetNonzeroElementsCount()
{
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 7x6 dense matrix:
    *
    *    /  0  2  3  4  5  6 \
    *    |  7  8  9 10 11 12 |
    *    | 13 14 15 16 17 18 |
    *    | 19 20 21 22 23 24 |
    *    | 25 26 27 28 29 30 |
    *    | 31 32 33 34 35 36 |
    *    \ 37 38 39 40 41  0 /
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   m.setElement( 0, 0, 0 );  // Set the first element of the diagonal to 0.
   m.setElement( 6, 5, 0 );  // Set the last element of the diagonal to 0.

   EXPECT_EQ( m.getNonzeroElementsCount(), 40 );
}

template< typename Matrix >
void
test_Reset()
{
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  0  0  0  0 \
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    \  0  0  0  0 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m;
   m.setDimensions( rows, cols );

   m.reset();

   EXPECT_EQ( m.getRows(), 0 );
   EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void
test_SetValue()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 7x6 dense matrix:
    *
    *    /  1  2  3  4  5  6 \
    *    |  7  8  9 10 11 12 |
    *    | 13 14 15 16 17 18 |
    *    | 19 20 21 22 23 24 |
    *    | 25 26 27 28 29 30 |
    *    | 31 32 33 34 35 36 |
    *    \ 37 38 39 40 41 42 /
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 4 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 5 } );
   EXPECT_EQ( m.getElement( 0, 5 ), RealType{ 6 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 7 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 8 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 9 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 10 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 1, 5 ), RealType{ 12 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 13 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 14 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 15 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 16 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 17 } );
   EXPECT_EQ( m.getElement( 2, 5 ), RealType{ 18 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 19 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 20 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 21 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 23 } );
   EXPECT_EQ( m.getElement( 3, 5 ), RealType{ 24 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 25 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 26 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 27 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 28 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 29 } );
   EXPECT_EQ( m.getElement( 4, 5 ), RealType{ 30 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 31 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 32 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 34 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 35 } );
   EXPECT_EQ( m.getElement( 5, 5 ), RealType{ 36 } );

   EXPECT_EQ( m.getElement( 6, 0 ), RealType{ 37 } );
   EXPECT_EQ( m.getElement( 6, 1 ), RealType{ 38 } );
   EXPECT_EQ( m.getElement( 6, 2 ), RealType{ 39 } );
   EXPECT_EQ( m.getElement( 6, 3 ), RealType{ 40 } );
   EXPECT_EQ( m.getElement( 6, 4 ), RealType{ 41 } );
   EXPECT_EQ( m.getElement( 6, 5 ), RealType{ 42 } );

   // Set the values of all elements to a certain number
   m.setValue( 42 );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 0, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 1, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 3, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 4, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 5, 5 ), RealType{ 42 } );

   EXPECT_EQ( m.getElement( 6, 0 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 6, 1 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 6, 2 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 6, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 6, 4 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 6, 5 ), RealType{ 42 } );
}

template< typename Matrix >
void
test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    \ 21 22 23 24 25 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 4 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 5 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 7 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 8 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 9 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 10 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 12 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 13 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 14 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 15 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 16 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 17 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 18 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 19 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 20 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 21 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 23 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 24 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 25 } );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > v( m.getRows(), 0 );
   auto v_view = v.getView();
   auto m_view = m.getView();
   auto f1 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      v_view[ i ] = m_view.getElement( i, i );
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, m.getRows(), f1 );

   for( IndexType i = 0; i < m.getRows(); i++ )
      EXPECT_EQ( v.getElement( i ), m.getElement( i, i ) );
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> bool
   {
      return ( v_view[ i ] == m_view.getElement( i, i ) );
   };
   EXPECT_TRUE( TNL::Algorithms::reduce< DeviceType >( (IndexType) 0, m.getRows(), fetch, std::logical_and<>{}, true ) );
}

template< typename Matrix >
void
test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    | 21 22 23 24 25 |
    *    \ 26 27 28 29 30 /
    */
   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 4 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 5 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 7 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 8 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 9 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 10 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 12 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 13 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 14 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 15 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 16 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 17 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 18 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 19 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 20 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 21 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 23 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 24 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 25 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 26 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 27 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 28 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 29 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 30 } );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 dense matrix:
    *
    *    /  3  6  9 12 15 \
    *    | 18 21 24 27 30 |
    *    | 33 36 39 42 45 |
    *    | 48 51 54 57 60 |
    *    | 63 66 69 72 75 |
    *    \ 78 81 84 87 90 /
    */
   IndexType newValue = 1;
   RealType multiplicator = 2;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.addElement( i, j, newValue++, multiplicator );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 9 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 12 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 15 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 18 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 21 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 24 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 27 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 30 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 36 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 39 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 42 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 45 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 48 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 51 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 54 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 57 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 60 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 63 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 66 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 69 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 72 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 75 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 78 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 81 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 84 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 87 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 90 } );
}

template< typename Matrix >
void
test_SetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 3x7 dense matrix:
    *
    *    / 11 11 11 11 11  6  7 \
    *    | 22 22 22 22 22 13 14 |
    *    \ 15 16 33 33 33 33 33 /
    */
   const IndexType rows = 3;
   const IndexType cols = 7;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   auto matrix_view = m.getView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      RealType values[ 3 ][ 5 ]{ { 11, 11, 11, 11, 11 }, { 22, 22, 22, 22, 22 }, { 33, 33, 33, 33, 33 } };
      IndexType columnIndexes[ 3 ][ 5 ]{ { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, { 2, 3, 4, 5, 6 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 5; i++ )
         row.setValue( columnIndexes[ rowIdx ][ i ], values[ rowIdx ][ i ] );
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, 3, f );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 5 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 0, 6 ), RealType{ 7 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 1, 5 ), RealType{ 13 } );
   EXPECT_EQ( m.getElement( 1, 6 ), RealType{ 14 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 15 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 16 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 2, 5 ), RealType{ 33 } );
   EXPECT_EQ( m.getElement( 2, 6 ), RealType{ 33 } );
}

template< typename Matrix >
void
test_AddRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    | 21 22 23 24 25 |
    *    \ 26 27 28 29 30 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 4 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 5 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 6 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 7 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 8 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 9 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 10 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 12 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 13 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 14 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 15 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 16 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 17 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 18 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 19 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 20 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 21 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 22 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 23 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 24 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 25 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 26 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 27 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 28 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 29 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 30 } );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *    /  3  6  9 12 15 \
    *    | 18 21 24 27 30 |
    *    | 33 36 39 42 45 |
    *    | 48 51 54 57 60 |
    *    | 63 66 69 72 75 |
    *    \ 78 81 84 87 90 /
    */

   auto matrix_view = m.getView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      RealType values[ 6 ][ 5 ]{ { 11, 11, 11, 11, 0 }, { 22, 22, 22, 22, 0 }, { 33, 33, 33, 33, 0 },
                                 { 44, 44, 44, 44, 0 }, { 55, 55, 55, 55, 0 }, { 66, 66, 66, 66, 0 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 5; i++ ) {
         RealType& val = row.getValue( i );
         val = RealType( rowIdx ) * val + values[ rowIdx ][ i ];
      }
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, 6, f );

   EXPECT_EQ( m.getElement( 0, 0 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 1 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 2 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 3 ), RealType{ 11 } );
   EXPECT_EQ( m.getElement( 0, 4 ), RealType{ 0 } );

   EXPECT_EQ( m.getElement( 1, 0 ), RealType{ 28 } );
   EXPECT_EQ( m.getElement( 1, 1 ), RealType{ 29 } );
   EXPECT_EQ( m.getElement( 1, 2 ), RealType{ 30 } );
   EXPECT_EQ( m.getElement( 1, 3 ), RealType{ 31 } );
   EXPECT_EQ( m.getElement( 1, 4 ), RealType{ 10 } );

   EXPECT_EQ( m.getElement( 2, 0 ), RealType{ 55 } );
   EXPECT_EQ( m.getElement( 2, 1 ), RealType{ 57 } );
   EXPECT_EQ( m.getElement( 2, 2 ), RealType{ 59 } );
   EXPECT_EQ( m.getElement( 2, 3 ), RealType{ 61 } );
   EXPECT_EQ( m.getElement( 2, 4 ), RealType{ 30 } );

   EXPECT_EQ( m.getElement( 3, 0 ), RealType{ 92 } );
   EXPECT_EQ( m.getElement( 3, 1 ), RealType{ 95 } );
   EXPECT_EQ( m.getElement( 3, 2 ), RealType{ 98 } );
   EXPECT_EQ( m.getElement( 3, 3 ), RealType{ 101 } );
   EXPECT_EQ( m.getElement( 3, 4 ), RealType{ 60 } );

   EXPECT_EQ( m.getElement( 4, 0 ), RealType{ 139 } );
   EXPECT_EQ( m.getElement( 4, 1 ), RealType{ 143 } );
   EXPECT_EQ( m.getElement( 4, 2 ), RealType{ 147 } );
   EXPECT_EQ( m.getElement( 4, 3 ), RealType{ 151 } );
   EXPECT_EQ( m.getElement( 4, 4 ), RealType{ 100 } );

   EXPECT_EQ( m.getElement( 5, 0 ), RealType{ 196 } );
   EXPECT_EQ( m.getElement( 5, 1 ), RealType{ 201 } );
   EXPECT_EQ( m.getElement( 5, 2 ), RealType{ 206 } );
   EXPECT_EQ( m.getElement( 5, 3 ), RealType{ 211 } );
   EXPECT_EQ( m.getElement( 5, 4 ), RealType{ 150 } );
}

template< typename Matrix >
void
test_ForElements()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x3 sparse matrix:
    *
    *    /  1  1  1  \
    *    |  2  2  2  |
    *    |  3  3  3  |
    *    |  4  4  4  |
    *    |  5  5  5  |
    *    |  6  6  6  |
    *    |  7  7  7  |
    *    \  8  8  8  /
    */

   const IndexType cols = 3;
   const IndexType rows = 8;

   Matrix m( rows, cols );
   m.forAllElements(
      [] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const IndexType& columnIdx, RealType& value ) mutable
      {
         value = rowIdx + 1.0;
      } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ )
         EXPECT_EQ( m.getElement( rowIdx, colIdx ), RealType( rowIdx + 1 ) );
}

template< typename Matrix >
void
test_ForRows()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   /////
   // Setup lower triangular matrix
   const IndexType cols = 8;
   const IndexType rows = 8;

   /////
   // Test without iterator
   Matrix m( rows, cols );
   using RowView = typename Matrix::RowView;
   m.forAllRows(
      [] __cuda_callable__( RowView & row ) mutable
      {
         for( IndexType localIdx = 0; localIdx <= row.getRowIndex(); localIdx++ )
            row.setValue( localIdx, row.getRowIndex() - localIdx + 1.0 );
      } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ ) {
         if( colIdx <= rowIdx )
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), RealType( rowIdx - colIdx + 1 ) );
         else
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), RealType{ 0 } );
      }

   /////
   // Test without iterator
   m.getValues() = 0.0;
   m.forAllRows(
      [] __cuda_callable__( RowView & row ) mutable
      {
         for( auto element : row )
            if( element.columnIndex() <= element.rowIndex() )
               element.value() = element.rowIndex() - element.columnIndex() + 1.0;
      } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ ) {
         if( colIdx <= rowIdx )
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), RealType( rowIdx - colIdx + 1 ) );
         else
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), RealType{ 0 } );
      }
}

template< typename Matrix >
void
test_reduceRows()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  4  5  0  1 \   6
    *    |  0  6  0  7  0  0  0  1 |   3
    *    |  0  8  9  0 10  0  0  1 |   4
    *    |  0 11 12 13 14  0  0  1 |   5
    *    |  0 15  0  0  0  0  0  1 |   2
    *    |  0 16 17 18 19 20 21  1 |   7
    *    | 22 23 24 25 26 27 28  1 |   8
    *    \ 29 30 31 32 33 34 35 36 /   8
    */

   const IndexType rows = 8;
   const IndexType cols = 8;

   Matrix m( { { 1, 2, 3, 0, 4, 5, 0, 1 },
               { 0, 6, 0, 7, 0, 0, 0, 1 },
               { 0, 8, 9, 0, 10, 0, 0, 1 },
               { 0, 11, 12, 13, 14, 0, 0, 1 },
               { 0, 15, 0, 0, 0, 0, 0, 1 },
               { 0, 16, 17, 18, 19, 20, 21, 1 },
               { 22, 23, 24, 25, 26, 27, 28, 1 },
               { 29, 30, 31, 32, 33, 34, 35, 36 } } );
   typename Matrix::RowCapacitiesType rowCapacities{ 6, 3, 4, 5, 2, 7, 8, 8 };

   IndexType value = 1;
   for( IndexType i = 0; i < 3; i++ )  // 0th row
      m.setElement( 0, i, value++ );

   m.setElement( 0, 4, value++ );  // 0th row
   m.setElement( 0, 5, value++ );

   m.setElement( 1, 1, value++ );  // 1st row
   m.setElement( 1, 3, value++ );

   for( IndexType i = 1; i < 3; i++ )  // 2nd row
      m.setElement( 2, i, value++ );

   m.setElement( 2, 4, value++ );  // 2nd row

   for( IndexType i = 1; i < 5; i++ )  // 3rd row
      m.setElement( 3, i, value++ );

   m.setElement( 4, 1, value++ );  // 4th row

   for( IndexType i = 1; i < 7; i++ )  // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 7; i++ )  // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 8; i++ )  // 7th row
      m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )  // 1s at the end of rows
      m.setElement( i, 7, 1 );

   ////
   // Compute number of non-zero elements in rows.
   typename Matrix::RowCapacitiesType rowLengths( rows );
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   m.reduceAllRows( fetch, std::plus<>{}, keep, 0 );
   EXPECT_EQ( rowCapacities, rowLengths );
   m.getCompressedRowLengths( rowLengths );
   EXPECT_EQ( rowCapacities, rowLengths );

   ////
   // Compute max norm
   using std::abs;
   using TNL::abs;
   using Real = decltype( abs( RealType{} ) );
   TNL::Containers::Vector< Real, DeviceType, IndexType > rowSums( rows );
   auto rowSums_view = rowSums.getView();
   auto max_fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return abs( value );
   };
   auto max_keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowSums_view[ rowIdx ] = value;
   };
   m.reduceAllRows( max_fetch, std::plus<>{}, max_keep, 0 );
   const Real maxNorm = TNL::max( rowSums );
   EXPECT_EQ( maxNorm, 260 );  // 29+30+31+32+33+34+35+36
}

template< typename Matrix >
void
test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    | 13 14 15 16 |
    *    \ 17 18 19 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   VectorType inVector;
   inVector.setSize( 4 );
   for( IndexType i = 0; i < inVector.getSize(); i++ )
      inVector.setElement( i, 2 );

   VectorType outVector;
   outVector.setSize( 5 );
   for( IndexType j = 0; j < outVector.getSize(); j++ )
      outVector.setElement( j, 0 );

   m.vectorProduct( inVector, outVector );

   EXPECT_EQ( outVector.getElement( 0 ), RealType{ 20 } );
   EXPECT_EQ( outVector.getElement( 1 ), RealType{ 52 } );
   EXPECT_EQ( outVector.getElement( 2 ), RealType{ 84 } );
   EXPECT_EQ( outVector.getElement( 3 ), RealType{ 116 } );
   EXPECT_EQ( outVector.getElement( 4 ), RealType{ 148 } );

   // Test for a very small matrix
   const Matrix A( { { 2.0, 5.0 }, { 1.0, 2.0 } } );

   VectorType x{ -2, 5 };

   VectorType result( x.getSize(), 0.0 );

   A.vectorProduct( x, result );

   EXPECT_EQ( result.getElement( 0 ), RealType{ 21 } );
   EXPECT_EQ( result.getElement( 1 ), RealType{ 8 } );
}

template< typename Matrix >
void
test_LargeVectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   if( std::is_same< IndexType, short >::value )
      return;

   const IndexType rows = 997;
   const IndexType cols = 997;

   Matrix m( rows, cols );
   m.forAllElements(
      [] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, RealType & value )
      {
         value = columnIdx + 1.0;
      } );

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   VectorType inVector( cols );
   inVector.forAllElements(
      [] __cuda_callable__( IndexType i, RealType & value )
      {
         value = 1.0;
      } );

   VectorType outVector( rows, 0.0 );

   m.vectorProduct( inVector, outVector );

   for( IndexType i = 0; i < rows; i++ ) {
      //RealType diag = ( i % 2 == 1 ? cols - 1 : -cols + 1 );
      //RealType non_diag = ( cols % 2 == 0 ? 0.0 : 1.0 );
      IndexType rcols = cols;
      EXPECT_EQ( outVector.getElement( i ), RealType( 0.5 * rcols * ( rcols + 1.0 ) ) );
   }
}

template< typename Matrix >
void
test_AddMatrix()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    | 13 14 15 16 |
    *    \ 17 18 19 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    | 13 14 15 16 |
    *    \ 17 18 19 20 /
    */

   Matrix m2;
   m2.reset();
   m2.setDimensions( rows, cols );

   IndexType newValue = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m2.setElement( i, j, newValue++ );

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    | 13 14 15 16 |
    *    \ 17 18 19 20 /
    */

   Matrix mResult;
   mResult.reset();
   mResult.setDimensions( rows, cols );

   mResult = m;

   RealType matrixMultiplicator = 2;
   RealType thisMatrixMultiplicator = 1;

   mResult.addMatrix( m2, matrixMultiplicator, thisMatrixMultiplicator );

   EXPECT_EQ( mResult.getElement( 0, 0 ),
              matrixMultiplicator * m2.getElement( 0, 0 ) + thisMatrixMultiplicator * m.getElement( 0, 0 ) );
   EXPECT_EQ( mResult.getElement( 0, 1 ),
              matrixMultiplicator * m2.getElement( 0, 1 ) + thisMatrixMultiplicator * m.getElement( 0, 1 ) );
   EXPECT_EQ( mResult.getElement( 0, 2 ),
              matrixMultiplicator * m2.getElement( 0, 2 ) + thisMatrixMultiplicator * m.getElement( 0, 2 ) );
   EXPECT_EQ( mResult.getElement( 0, 3 ),
              matrixMultiplicator * m2.getElement( 0, 3 ) + thisMatrixMultiplicator * m.getElement( 0, 3 ) );

   EXPECT_EQ( mResult.getElement( 1, 0 ),
              matrixMultiplicator * m2.getElement( 1, 0 ) + thisMatrixMultiplicator * m.getElement( 1, 0 ) );
   EXPECT_EQ( mResult.getElement( 1, 1 ),
              matrixMultiplicator * m2.getElement( 1, 1 ) + thisMatrixMultiplicator * m.getElement( 1, 1 ) );
   EXPECT_EQ( mResult.getElement( 1, 2 ),
              matrixMultiplicator * m2.getElement( 1, 2 ) + thisMatrixMultiplicator * m.getElement( 1, 2 ) );
   EXPECT_EQ( mResult.getElement( 1, 3 ),
              matrixMultiplicator * m2.getElement( 1, 3 ) + thisMatrixMultiplicator * m.getElement( 1, 3 ) );

   EXPECT_EQ( mResult.getElement( 2, 0 ),
              matrixMultiplicator * m2.getElement( 2, 0 ) + thisMatrixMultiplicator * m.getElement( 2, 0 ) );
   EXPECT_EQ( mResult.getElement( 2, 1 ),
              matrixMultiplicator * m2.getElement( 2, 1 ) + thisMatrixMultiplicator * m.getElement( 2, 1 ) );
   EXPECT_EQ( mResult.getElement( 2, 2 ),
              matrixMultiplicator * m2.getElement( 2, 2 ) + thisMatrixMultiplicator * m.getElement( 2, 2 ) );
   EXPECT_EQ( mResult.getElement( 2, 3 ),
              matrixMultiplicator * m2.getElement( 2, 3 ) + thisMatrixMultiplicator * m.getElement( 2, 3 ) );

   EXPECT_EQ( mResult.getElement( 3, 0 ),
              matrixMultiplicator * m2.getElement( 3, 0 ) + thisMatrixMultiplicator * m.getElement( 3, 0 ) );
   EXPECT_EQ( mResult.getElement( 3, 1 ),
              matrixMultiplicator * m2.getElement( 3, 1 ) + thisMatrixMultiplicator * m.getElement( 3, 1 ) );
   EXPECT_EQ( mResult.getElement( 3, 2 ),
              matrixMultiplicator * m2.getElement( 3, 2 ) + thisMatrixMultiplicator * m.getElement( 3, 2 ) );
   EXPECT_EQ( mResult.getElement( 3, 3 ),
              matrixMultiplicator * m2.getElement( 3, 3 ) + thisMatrixMultiplicator * m.getElement( 3, 3 ) );

   EXPECT_EQ( mResult.getElement( 4, 0 ),
              matrixMultiplicator * m2.getElement( 4, 0 ) + thisMatrixMultiplicator * m.getElement( 4, 0 ) );
   EXPECT_EQ( mResult.getElement( 4, 1 ),
              matrixMultiplicator * m2.getElement( 4, 1 ) + thisMatrixMultiplicator * m.getElement( 4, 1 ) );
   EXPECT_EQ( mResult.getElement( 4, 2 ),
              matrixMultiplicator * m2.getElement( 4, 2 ) + thisMatrixMultiplicator * m.getElement( 4, 2 ) );
   EXPECT_EQ( mResult.getElement( 4, 3 ),
              matrixMultiplicator * m2.getElement( 4, 3 ) + thisMatrixMultiplicator * m.getElement( 4, 3 ) );

   EXPECT_EQ( mResult.getElement( 0, 0 ), RealType{ 3 } );
   EXPECT_EQ( mResult.getElement( 0, 1 ), RealType{ 6 } );
   EXPECT_EQ( mResult.getElement( 0, 2 ), RealType{ 9 } );
   EXPECT_EQ( mResult.getElement( 0, 3 ), RealType{ 12 } );

   EXPECT_EQ( mResult.getElement( 1, 0 ), RealType{ 15 } );
   EXPECT_EQ( mResult.getElement( 1, 1 ), RealType{ 18 } );
   EXPECT_EQ( mResult.getElement( 1, 2 ), RealType{ 21 } );
   EXPECT_EQ( mResult.getElement( 1, 3 ), RealType{ 24 } );

   EXPECT_EQ( mResult.getElement( 2, 0 ), RealType{ 27 } );
   EXPECT_EQ( mResult.getElement( 2, 1 ), RealType{ 30 } );
   EXPECT_EQ( mResult.getElement( 2, 2 ), RealType{ 33 } );
   EXPECT_EQ( mResult.getElement( 2, 3 ), RealType{ 36 } );

   EXPECT_EQ( mResult.getElement( 3, 0 ), RealType{ 39 } );
   EXPECT_EQ( mResult.getElement( 3, 1 ), RealType{ 42 } );
   EXPECT_EQ( mResult.getElement( 3, 2 ), RealType{ 45 } );
   EXPECT_EQ( mResult.getElement( 3, 3 ), RealType{ 48 } );

   EXPECT_EQ( mResult.getElement( 4, 0 ), RealType{ 51 } );
   EXPECT_EQ( mResult.getElement( 4, 1 ), RealType{ 54 } );
   EXPECT_EQ( mResult.getElement( 4, 2 ), RealType{ 57 } );
   EXPECT_EQ( mResult.getElement( 4, 3 ), RealType{ 60 } );
}

template< typename Matrix >
void
test_GetMatrixProduct()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    | 13 14 15 16 |
    *    \ 17 18 19 20 /
    */
   const IndexType leftRows = 5;
   const IndexType leftCols = 4;

   Matrix leftMatrix;
   leftMatrix.setDimensions( leftRows, leftCols );

   IndexType value = 1;
   for( IndexType i = 0; i < leftRows; i++ )
      for( IndexType j = 0; j < leftCols; j++ )
         leftMatrix.setElement( i, j, value++ );

   /*
    * Sets up the following 4x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    \ 16 17 18 19 20 /
    */
   const IndexType rightRows = 4;
   const IndexType rightCols = 5;

   Matrix rightMatrix;
   rightMatrix.setDimensions( rightRows, rightCols );

   IndexType newValue = 1;
   for( IndexType i = 0; i < rightRows; i++ )
      for( IndexType j = 0; j < rightCols; j++ )
         rightMatrix.setElement( i, j, newValue++ );

   /*
    * Perform the matrix multiplication:
    *
    *      /  1  2  3  4 \                        /  220  240  260  280  300 \
    *      |  5  6  7  8 |   /  1  2  3  4  5 \   |  492  544  596  648  700 |
    *  2 * |  9 10 11 12 | * |  6  7  8  9 10 | = |  764  848  932 1016 1100 |
    *      | 13 14 15 16 |   | 11 12 13 14 15 |   | 1036 1152 1268 1384 1500 |
    *      \ 17 18 19 20 /   \ 16 17 18 19 20 /   \ 1308 1456 1604 1752 1900 /
    */
   Matrix mResult;
   mResult.setValue( 0 );
   RealType matrixMultiplicator = 2;
   mResult.getMatrixProduct( leftMatrix, rightMatrix, matrixMultiplicator );

   EXPECT_EQ( mResult.getElement( 0, 0 ), RealType{ 220 } );
   EXPECT_EQ( mResult.getElement( 0, 1 ), RealType{ 240 } );
   EXPECT_EQ( mResult.getElement( 0, 2 ), RealType{ 260 } );
   EXPECT_EQ( mResult.getElement( 0, 3 ), RealType{ 280 } );
   EXPECT_EQ( mResult.getElement( 0, 4 ), RealType{ 300 } );

   EXPECT_EQ( mResult.getElement( 1, 0 ), RealType{ 492 } );
   EXPECT_EQ( mResult.getElement( 1, 1 ), RealType{ 544 } );
   EXPECT_EQ( mResult.getElement( 1, 2 ), RealType{ 596 } );
   EXPECT_EQ( mResult.getElement( 1, 3 ), RealType{ 648 } );
   EXPECT_EQ( mResult.getElement( 1, 4 ), RealType{ 700 } );

   EXPECT_EQ( mResult.getElement( 2, 0 ), RealType{ 764 } );
   EXPECT_EQ( mResult.getElement( 2, 1 ), RealType{ 848 } );
   EXPECT_EQ( mResult.getElement( 2, 2 ), RealType{ 932 } );
   EXPECT_EQ( mResult.getElement( 2, 3 ), RealType{ 1016 } );
   EXPECT_EQ( mResult.getElement( 2, 4 ), RealType{ 1100 } );

   EXPECT_EQ( mResult.getElement( 3, 0 ), RealType{ 1036 } );
   EXPECT_EQ( mResult.getElement( 3, 1 ), RealType{ 1152 } );
   EXPECT_EQ( mResult.getElement( 3, 2 ), RealType{ 1268 } );
   EXPECT_EQ( mResult.getElement( 3, 3 ), RealType{ 1384 } );
   EXPECT_EQ( mResult.getElement( 3, 4 ), RealType{ 1500 } );

   EXPECT_EQ( mResult.getElement( 4, 0 ), RealType{ 1308 } );
   EXPECT_EQ( mResult.getElement( 4, 1 ), RealType{ 1456 } );
   EXPECT_EQ( mResult.getElement( 4, 2 ), RealType{ 1604 } );
   EXPECT_EQ( mResult.getElement( 4, 3 ), RealType{ 1752 } );
   EXPECT_EQ( mResult.getElement( 4, 4 ), RealType{ 1900 } );
}

template< typename Matrix >
void
test_GetTransposition()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 3x2 dense matrix:
    *
    *    /  1  2 \
    *    |  3  4 |
    *    \  5  6 /
    */
   const IndexType rows = 3;
   const IndexType cols = 2;

   Matrix m;
   m.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   // Compute the transposition
   Matrix mTransposed;
   RealType matrixMultiplicator = 1;
   mTransposed.getTransposition( m, matrixMultiplicator );

   /*
    * Should result in the following 2x3 dense matrix:
    *
    *    /  1  3  5 \
    *    \  2  4  6 /
    */

   EXPECT_EQ( mTransposed.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( mTransposed.getElement( 0, 1 ), RealType{ 3 } );
   EXPECT_EQ( mTransposed.getElement( 0, 2 ), RealType{ 5 } );

   EXPECT_EQ( mTransposed.getElement( 1, 0 ), RealType{ 2 } );
   EXPECT_EQ( mTransposed.getElement( 1, 1 ), RealType{ 4 } );
   EXPECT_EQ( mTransposed.getElement( 1, 2 ), RealType{ 6 } );
}

template< typename Matrix >
void
test_AssignmentOperator()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   using DenseHost = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType >;

   const IndexType rows( 10 ), columns( 10 );
   DenseHost hostMatrix( rows, columns );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j <= i; j++ )
         hostMatrix( i, j ) = i + j;

   Matrix matrix( rows, columns );
   matrix.getValues() = 0.0;
   matrix = hostMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ ) {
         if( j > i )
            EXPECT_EQ( matrix.getElement( i, j ), RealType{ 0 } );
         else
            EXPECT_EQ( matrix.getElement( i, j ), RealType( i + j ) );
      }

#ifdef __CUDACC__
   using DenseCuda = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Cuda, IndexType >;
   DenseCuda cudaMatrix( rows, columns );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j <= i; j++ )
         cudaMatrix.setElement( i, j, i + j );

   matrix.getValues() = 0.0;
   matrix = cudaMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ ) {
         if( j > i )
            EXPECT_EQ( matrix.getElement( i, j ), RealType{ 0 } );
         else
            EXPECT_EQ( matrix.getElement( i, j ), RealType( i + j ) );
      }
#endif
}

template< typename Matrix >
void
test_SaveAndLoad()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 4x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    \ 13 14 15 16 /
    */
   const IndexType rows = 4;
   const IndexType cols = 4;

   Matrix savedMatrix;
   savedMatrix.reset();
   savedMatrix.setDimensions( rows, cols );

   IndexType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         savedMatrix.setElement( i, j, value++ );

   ASSERT_NO_THROW( savedMatrix.save( TEST_FILE_NAME ) );

   Matrix loadedMatrix;

   ASSERT_NO_THROW( loadedMatrix.load( TEST_FILE_NAME ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ), loadedMatrix.getElement( 0, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ), loadedMatrix.getElement( 0, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ), loadedMatrix.getElement( 0, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ), loadedMatrix.getElement( 0, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ), loadedMatrix.getElement( 1, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ), loadedMatrix.getElement( 1, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ), loadedMatrix.getElement( 1, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ), loadedMatrix.getElement( 1, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ), loadedMatrix.getElement( 2, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), loadedMatrix.getElement( 2, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), loadedMatrix.getElement( 2, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), loadedMatrix.getElement( 2, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ), loadedMatrix.getElement( 3, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ), loadedMatrix.getElement( 3, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), loadedMatrix.getElement( 3, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), loadedMatrix.getElement( 3, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ), RealType{ 1 } );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ), RealType{ 2 } );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ), RealType{ 3 } );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ), RealType{ 4 } );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ), RealType{ 5 } );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ), RealType{ 6 } );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ), RealType{ 7 } );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ), RealType{ 8 } );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ), RealType{ 9 } );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), RealType{ 10 } );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), RealType{ 11 } );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), RealType{ 12 } );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ), RealType{ 13 } );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ), RealType{ 14 } );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), RealType{ 15 } );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), RealType{ 16 } );
}

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, short, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Host, short, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, short, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, short, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >
#ifdef __CUDACC__
   ,
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< long, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
   // TNL::Matrices::DenseMatrix< TNL::Arithmetics::Complex<float>,  TNL::Devices::Cuda, long,
   // TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >
#endif
   >;

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, getSerializationType )
{
   test_GetSerializationType();
}

TYPED_TEST( MatrixTest, setDimensionsTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetDimensions< MatrixType >();
}

TYPED_TEST( MatrixTest, setLikeTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST( MatrixTest, setElementsTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetElements< MatrixType >();
}

TYPED_TEST( MatrixTest, getAllocatedElementsCountTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_GetAllocatedElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, getNonzeroElementsCountTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_GetNonzeroElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, resetTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Reset< MatrixType >();
}

TYPED_TEST( MatrixTest, setValueTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetValue< MatrixType >();
}

TYPED_TEST( MatrixTest, setElementTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetElement< MatrixType >();
}

TYPED_TEST( MatrixTest, addElementTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_AddElement< MatrixType >();
}

TYPED_TEST( MatrixTest, setRowTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetRow< MatrixType >();
}

TYPED_TEST( MatrixTest, addRowTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_AddRow< MatrixType >();
}

TYPED_TEST( MatrixTest, forElementsTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_ForElements< MatrixType >();
}

TYPED_TEST( MatrixTest, forRowsTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_ForRows< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_VectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, largeVectorProductTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_LargeVectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, addMatrixTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_AddMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, getMatrixProductTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_GetMatrixProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, getTranspositionTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_GetTransposition< MatrixType >();
}

TYPED_TEST( MatrixTest, assignmentOperatorTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_AssignmentOperator< MatrixType >();
}

TYPED_TEST( MatrixTest, saveAndLoadTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SaveAndLoad< MatrixType >();
}

#include "../main.h"
