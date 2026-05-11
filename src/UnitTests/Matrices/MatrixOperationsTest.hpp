#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

template< typename TripleType >
void
getDiagonal_DenseMatrix_test()
{
   using RealType = std::tuple_element_t< 0, TripleType >;
   using DeviceType = std::tuple_element_t< 1, TripleType >;
   using IndexType = std::tuple_element_t< 2, TripleType >;

   //using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   using MatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;
   // Test square matrix
   {
      MatrixType matrix( 4, 4 );
      matrix.setElement( 0, 0, 1.0 );
      matrix.setElement( 1, 1, 2.0 );
      matrix.setElement( 2, 2, 3.0 );
      matrix.setElement( 3, 3, 4.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
      EXPECT_EQ( diagonal.getElement( 3 ), 4.0 );
   }

   // Test rectangular matrix (rows > cols)
   {
      MatrixType matrix( 5, 3 );
      matrix.setElement( 0, 0, 1.0 );
      matrix.setElement( 1, 1, 2.0 );
      matrix.setElement( 2, 2, 3.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
   }

   // Test rectangular matrix (cols > rows)
   {
      MatrixType matrix( 3, 5 );
      matrix.setElement( 0, 0, 1.0 );
      matrix.setElement( 1, 1, 2.0 );
      matrix.setElement( 2, 2, 3.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
   }

   // Test square matrix with all elements set
   {
      MatrixType matrix( 4, 4 );
      for( IndexType i = 0; i < 4; i++ )
         for( IndexType j = 0; j < 4; j++ )
            matrix.setElement( i, j, i * 4 + j + 1.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );  // matrix(0,0)
      EXPECT_EQ( diagonal.getElement( 1 ), 6.0 );  // matrix(1,1)
      EXPECT_EQ( diagonal.getElement( 2 ), 11.0 );  // matrix(2,2)
      EXPECT_EQ( diagonal.getElement( 3 ), 16.0 );  // matrix(3,3)
   }

   // Test rectangular matrix with all elements set (rows > cols)
   {
      MatrixType matrix( 4, 3 );
      for( IndexType i = 0; i < 4; i++ )
         for( IndexType j = 0; j < 3; j++ )
            matrix.setElement( i, j, i * 3 + j + 1.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );  // matrix(0,0)
      EXPECT_EQ( diagonal.getElement( 1 ), 5.0 );  // matrix(1,1)
      EXPECT_EQ( diagonal.getElement( 2 ), 9.0 );  // matrix(2,2)
   }

   // Test rectangular matrix with all elements set (cols > rows)
   {
      MatrixType matrix( 3, 4 );
      for( IndexType i = 0; i < 3; i++ )
         for( IndexType j = 0; j < 4; j++ )
            matrix.setElement( i, j, i * 4 + j + 1.0 );

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );  // matrix(0,0)
      EXPECT_EQ( diagonal.getElement( 1 ), 6.0 );  // matrix(1,1)
      EXPECT_EQ( diagonal.getElement( 2 ), 11.0 );  // matrix(2,2)
   }
}
template< typename TripleType >
void
getDiagonal_SparseMatrix_test()
{
   using RealType = std::tuple_element_t< 0, TripleType >;
   using DeviceType = std::tuple_element_t< 1, TripleType >;
   using IndexType = std::tuple_element_t< 2, TripleType >;

   using MatrixType = TNL::Matrices::SparseMatrix< RealType, DeviceType, IndexType >;

   // Test square matrix
   {
      // clang-format off
      MatrixType matrix( 4, 4, {
         { 0, 0, 1.0 },
                       { 1, 1, 2.0 },
                                      { 2, 2, 3.0 },
         { 3, 0, 1.0 },                              { 3, 3, 4.0 } } );
      // clang-format on

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
      EXPECT_EQ( diagonal.getElement( 3 ), 4.0 );
   }

   // Test rectangular matrix (rows > cols)
   {
      // clang-format off
      MatrixType matrix( 5, 3, {
         { 0, 0, 1.0 },                { 0, 2, 3.0 }, 
                        { 1, 1, 2.0 },
         { 2, 0, 2.0 },                { 2, 2, 3.0 },
         { 3, 0, 1.0 },                { 3, 2, 4.0 }, 
                        { 4, 1, 1.0 }, { 4, 2, 4.0 } 
      } );
      // clang-format on

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
   }

   // Test rectangular matrix (cols > rows)
   {
      // clang-format off
      MatrixType matrix( 3, 5, {
         { 0, 0, 1.0 },               { 0, 2, 3.0 },                { 0, 4, 5.0 },
                        { 1, 1, 2.0 },               { 1, 3, 4.0 },
         { 2, 0, 2.0 },               { 2, 2, 3.0 },                { 2, 4, 6.0 }
      } );
      // clang-format on

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 2.0 );
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
   }

   // Test sparse matrix with missing diagonal elements
   {
      // clang-format off
      MatrixType matrix( 4, 4, {
         { 0, 0, 1.0 },
         { 0, 1, 5.0 },
                               { 2, 2, 3.0 },
                                              { 3, 3, 4.0 } } );
      // clang-format on

      auto diagonal = TNL::Matrices::getDiagonal( matrix );

      EXPECT_EQ( diagonal.getElement( 0 ), 1.0 );
      EXPECT_EQ( diagonal.getElement( 1 ), 0.0 );  // missing diagonal element
      EXPECT_EQ( diagonal.getElement( 2 ), 3.0 );
      EXPECT_EQ( diagonal.getElement( 3 ), 4.0 );
   }
}
