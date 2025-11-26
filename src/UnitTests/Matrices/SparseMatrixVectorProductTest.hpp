#pragma once

#include <functional>
#include <iostream>
#include <sstream>

#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>

#include <gtest/gtest.h>

template< typename Matrix >
void
test_VectorProduct_zeroMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up an empty 4x4 sparse matrix with the following row capacities: [1, 2, 1, 1].
    * The matrix values are uninitialized and the column indexes are set to the padding index (-1).
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1;
   m_1.reset();
   m_1.setDimensions( m_rows_1, m_cols_1 );
   typename Matrix::RowCapacitiesType rowLengths_1{ 1, 2, 1, 1 };
   m_1.setRowCapacities( rowLengths_1 );

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_1.getSegments() ) ) {
      SCOPED_TRACE( tag );

      VectorType inVector_1;
      inVector_1.setSize( m_cols_1 );
      inVector_1.setValue( 1 );

      VectorType outVector_1;
      outVector_1.setSize( m_rows_1 );
      outVector_1.setValue( -1 );

      m_1.vectorProduct( inVector_1, outVector_1, launch_config );

      EXPECT_EQ( outVector_1.getElement( 0 ), RealType{ 0 } );
      EXPECT_EQ( outVector_1.getElement( 1 ), RealType{ 0 } );
      EXPECT_EQ( outVector_1.getElement( 2 ), RealType{ 0 } );
      EXPECT_EQ( outVector_1.getElement( 3 ), RealType{ 0 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_1_transposed;
      m_1_transposed.getTransposition( m_1 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_1_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_1, 1 );
         VectorType outVector_1_transposed( m_cols_1, -1 );
         VectorType outVector_2_transposed( m_cols_1, -1 );
         m_1_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_1.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
   }
}

template< typename Matrix >
void
test_VectorProduct_smallMatrix1()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

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

   // clang-format off
   Matrix m_1( m_rows_1, m_cols_1,
   { {0,0,1},
              {1,1,2},          {1,3,3},
              {2,1,4},
                       {3,2,5} } );
   // clang-format on

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_1.getSegments() ) ) {
      SCOPED_TRACE( tag );

      VectorType inVector_1( m_cols_1, 2 );
      VectorType outVector_1( m_rows_1, 0 );

      m_1.vectorProduct( inVector_1, outVector_1, launch_config );

      EXPECT_EQ( outVector_1.getElement( 0 ), RealType{ 2 } );
      EXPECT_EQ( outVector_1.getElement( 1 ), RealType{ 10 } );
      EXPECT_EQ( outVector_1.getElement( 2 ), RealType{ 8 } );
      EXPECT_EQ( outVector_1.getElement( 3 ), RealType{ 10 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_1_transposed;
      m_1_transposed.getTransposition( m_1 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_1_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_1, 1.0 );
         VectorType outVector_1_transposed( m_cols_1, 0.0 );
         VectorType outVector_2_transposed( m_cols_1, 0.0 );
         m_1_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_1.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
   }
}

template< typename Matrix >
void
test_VectorProduct_smallMatrix2()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

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

   // clang-format off
   Matrix m_2( m_rows_2, m_cols_2,
   { {0,0,1}, {0,1,2}, {0,2,3},
              {1,3,4},
     {2,0,5}, {2,1,6}, {2,2,7},
              {3,1,8} } );
   // clang-format on

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_2.getSegments() ) ) {
      SCOPED_TRACE( tag );
      VectorType inVector_2( m_cols_2, 2 );
      VectorType outVector_2( m_rows_2, 0 );

      m_2.vectorProduct( inVector_2, outVector_2, launch_config );

      EXPECT_EQ( outVector_2.getElement( 0 ), RealType{ 12 } );
      EXPECT_EQ( outVector_2.getElement( 1 ), RealType{ 8 } );
      EXPECT_EQ( outVector_2.getElement( 2 ), RealType{ 36 } );
      EXPECT_EQ( outVector_2.getElement( 3 ), RealType{ 16 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_2_transposed;
      m_2_transposed.getTransposition( m_2 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_2_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_2, 1.0 );
         VectorType outVector_1_transposed( m_cols_2, 0.0 );
         VectorType outVector_2_transposed( m_cols_2, 0.0 );
         m_2_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_2.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
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

   //typename Matrix::RowCapacitiesType rowLengths_3{ 3, 3, 3, 3 };
   //m_3.setRowCapacities( rowLengths_3 );

   // clang-format off
   Matrix m_3( m_rows_3, m_cols_3,
   { {0,0,1}, {0,1,2},  {0,2,3},
              {1,1,4},  {1,2,5},  {1,3,6},
     {2,0,7}, {2,1,8},  {2,2,9},
              {3,1,10}, {3,2,11}, {3,3,12} } );
   // clang-format on

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_3.getSegments() ) ) {
      SCOPED_TRACE( tag );

      VectorType inVector_3( m_cols_3, 2 );
      VectorType outVector_3( m_rows_3, 0 );

      m_3.vectorProduct( inVector_3, outVector_3, launch_config );

      EXPECT_EQ( outVector_3.getElement( 0 ), RealType{ 12 } );
      EXPECT_EQ( outVector_3.getElement( 1 ), RealType{ 30 } );
      EXPECT_EQ( outVector_3.getElement( 2 ), RealType{ 48 } );
      EXPECT_EQ( outVector_3.getElement( 3 ), RealType{ 66 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_3_transposed;
      m_3_transposed.getTransposition( m_3 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_3_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_3, 1.0 );
         VectorType outVector_1_transposed( m_cols_3, 0.0 );
         VectorType outVector_2_transposed( m_cols_3, 0.0 );
         m_3_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_3.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
   }
}

template< typename Matrix >
void
test_VectorProduct_mediumSizeMatrix1()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  0  4  0  0 \
    *    |  0  5  6  7  8  0  0  0 |
    *    |  9 10 11 12 13  0  0  0 |
    *    |  0 14 15 16 17  0  0  0 |
    *    |  0  0 18 19 20 21  0  0 |
    *    |  0  0  0 22 23 24 25  0 |
    *    | 26 27 28 29 30  0  0  0 |
    *    \ 31 32 33 34 35  0  0  0 /
    */

   const IndexType m_rows_4 = 8;
   const IndexType m_cols_4 = 8;

   //typename Matrix::RowCapacitiesType rowLengths_4{ 4, 4, 5, 4, 4, 4, 5, 5 };
   //m_4.setRowCapacities( rowLengths_4 );

   // clang-format off
   Matrix m_4( m_rows_4, m_cols_4,
   { {0,0,1},  {0,1,2},  {0,2,3},                      {0,5,4},
               {1,1,5},  {1,2,6},  {1,3,7},  {1,4,8},
     {2,0,9},  {2,1,10}, {2,2,11}, {2,3,12}, {2,4,13},
               {3,1,14}, {3,2,15}, {3,3,16}, {3,4,17},
                         {4,2,18}, {4,3,19}, {4,4,20}, {4,5,21},
                                   {5,3,22}, {5,4,23}, {5,5,24}, {5,6,25},
     {6,0,26}, {6,1,27}, {6,2,28}, {6,3,29}, {6,4,30},
     {7,0,31}, {7,1,32}, {7,2,33}, {7,3,34}, {7,4 ,35} } );
   // clang-format on

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_4.getSegments() ) ) {
      SCOPED_TRACE( tag );

      VectorType inVector_4( m_cols_4, 2 );
      VectorType outVector_4( m_rows_4, 0 );

      m_4.vectorProduct( inVector_4, outVector_4, launch_config );

      EXPECT_EQ( outVector_4.getElement( 0 ), RealType{ 20 } );
      EXPECT_EQ( outVector_4.getElement( 1 ), RealType{ 52 } );
      EXPECT_EQ( outVector_4.getElement( 2 ), RealType{ 110 } );
      EXPECT_EQ( outVector_4.getElement( 3 ), RealType{ 124 } );
      EXPECT_EQ( outVector_4.getElement( 4 ), RealType{ 156 } );
      EXPECT_EQ( outVector_4.getElement( 5 ), RealType{ 188 } );
      EXPECT_EQ( outVector_4.getElement( 6 ), RealType{ 280 } );
      EXPECT_EQ( outVector_4.getElement( 7 ), RealType{ 330 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_4_transposed;
      m_4_transposed.getTransposition( m_4 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_4_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_4, 1.0 );
         VectorType outVector_1_transposed( m_cols_4, 0.0 );
         VectorType outVector_2_transposed( m_cols_4, 0.0 );
         m_4_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_4.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
   }
}

template< typename Matrix >
void
test_VectorProduct_mediumSizeMatrix2()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

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

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   // clang-format off
   Matrix m_5( m_rows_5, m_cols_5,
   { {0,0, 1}, {0,1, 2}, {0,2, 3},           {0,4,4},  {0,5, 5},           {0,7, 1},
               {1,1, 6},           {1,3, 7},                               {1,7, 1},
               {2,1, 8}, {2,2, 9},           {2,4,10},                     {2,7, 1},
               {3,1,11}, {3,2,12}, {3,3,13}, {3,4,14},                     {3,7, 1},
               {4,1,15},                                                   {4,7, 1},
               {5,1,16}, {5,2,17}, {5,3,18}, {5,4,19}, {5,5,20}, {5,6,21}, {5,7, 1},
     {6,0,22}, {6,1,23}, {6,2,24}, {6,3,25}, {6,4,26}, {6,5,27}, {6,6,28}, {6,7, 1},
     {7,0,29}, {7,1,30}, {7,2,31}, {7,3,32}, {7,4,33}, {7,5,34}, {7,6,35}, {7,7,36} } );
   // clang-format on

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m_5.getSegments() ) ) {
      SCOPED_TRACE( tag );

      VectorType inVector_5( m_cols_5, 2 );
      VectorType outVector_5( m_rows_5, 0 );

      m_5.vectorProduct( inVector_5, outVector_5, launch_config );

      EXPECT_EQ( outVector_5.getElement( 0 ), RealType{ 32 } );
      EXPECT_EQ( outVector_5.getElement( 1 ), RealType{ 28 } );
      EXPECT_EQ( outVector_5.getElement( 2 ), RealType{ 56 } );
      EXPECT_EQ( outVector_5.getElement( 3 ), RealType{ 102 } );
      EXPECT_EQ( outVector_5.getElement( 4 ), RealType{ 32 } );
      EXPECT_EQ( outVector_5.getElement( 5 ), RealType{ 224 } );
      EXPECT_EQ( outVector_5.getElement( 6 ), RealType{ 352 } );
      EXPECT_EQ( outVector_5.getElement( 7 ), RealType{ 520 } );
   }

   // Test transposedVectorProduct
   // TODO: implement it for complex types
   if constexpr( ! TNL::is_complex_v< RealType > ) {
      Matrix m_5_transposed;
      m_5_transposed.getTransposition( m_5 );
      for( auto [ launch_config, tag ] :
           TNL::Algorithms::Segments::reductionLaunchConfigurations( m_5_transposed.getSegments() ) )
      {
         SCOPED_TRACE( tag );

         VectorType inVector_1_transposed( m_rows_5, 1.0 );
         VectorType outVector_1_transposed( m_cols_5, 0.0 );
         VectorType outVector_2_transposed( m_cols_5, 0.0 );
         m_5_transposed.vectorProduct( inVector_1_transposed, outVector_1_transposed, launch_config );
         m_5.transposedVectorProduct( inVector_1_transposed, outVector_2_transposed );
         EXPECT_EQ( outVector_1_transposed, outVector_2_transposed );
      }
   }
}

template< typename Matrix >
void
test_VectorProduct_largeMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using OutRealType = std::conditional_t< TNL::is_complex_v< RealType >, RealType, double >;

   const IndexType size( 1051 );

   // Test with large diagonal matrix
   Matrix m1( size, size );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowCapacities( size );
   rowCapacities.forAllElements(
      [] __cuda_callable__( IndexType i, IndexType & value )
      {
         value = 1;
      } );
   m1.setRowCapacities( rowCapacities );
   auto f1 = [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType & column, RealType & value )
   {
      if( localIdx == 0 ) {
         value = row + 1;
         column = row;
      }
   };
   m1.forAllElements( f1 );
   // check that the matrix was initialized
   m1.getCompressedRowLengths( rowCapacities );
   EXPECT_EQ( rowCapacities, 1 );

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m1.getSegments() ) ) {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< OutRealType, DeviceType, IndexType > in( size, 1.0 ), out( size, 0.0 );
      m1.vectorProduct( in, out, launch_config );
      for( IndexType i = 0; i < size; i++ )
         EXPECT_EQ( out.getElement( i ), OutRealType( i + 1 ) );
   }

   // Test with large triangular matrix
   const int rows( size ), columns( size );
   Matrix m2( rows, columns );
   rowCapacities.setSize( rows );
   rowCapacities.forAllElements(
      [ = ] __cuda_callable__( IndexType i, IndexType & value )
      {
         value = i + 1;
      } );
   m2.setRowCapacities( rowCapacities );
   auto f2 = [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType & column, RealType & value )
   {
      if( localIdx <= row ) {
         value = localIdx + 1;
         column = localIdx;
      }
   };
   m2.forAllElements( f2 );
   // check that the matrix was initialized
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowLengths( rows );
   m2.getCompressedRowLengths( rowLengths );
   EXPECT_EQ( rowLengths, rowCapacities );

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m2.getSegments() ) ) {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< OutRealType, DeviceType, IndexType > in( size, 1.0 ), out( size, 0.0 );
      m2.vectorProduct( in, out, launch_config );
      for( IndexType i = 0; i < rows; i++ )
         EXPECT_EQ( out.getElement( i ), OutRealType( ( i + 1 ) * ( i + 2 ) / 2 ) ) << " at row " << i;
   }
}

template< typename Matrix >
void
test_VectorProduct_longRowsMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using OutRealType = std::conditional_t< TNL::is_complex_v< RealType >, RealType, double >;

   /**
    * Long row test
    */
   for( auto columns : { 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 3000 } ) {
      const int rows = 33;
      Matrix m3( rows, columns );
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowCapacities( rows );
      rowCapacities = columns;
      m3.setRowCapacities( rowCapacities );
      auto f = [] __cuda_callable__( IndexType row, IndexType localIdx, IndexType & column, RealType & value )
      {
         column = localIdx;
         value = localIdx + row;
      };
      m3.forAllElements( f );
      for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( m3.getSegments() ) ) {
         SCOPED_TRACE( tag );

         TNL::Containers::Vector< OutRealType, DeviceType, IndexType > in( columns, 1.0 ), out( rows, 0.0 );
         m3.vectorProduct( in, out, launch_config );
         for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
            EXPECT_EQ( out.getElement( rowIdx ), OutRealType( columns * ( columns - 1 ) / 2.0 + columns * rowIdx ) );
      }
   }
}
