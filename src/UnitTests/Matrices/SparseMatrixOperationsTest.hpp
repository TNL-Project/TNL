#pragma once

#include <functional>
#include <iostream>
#include <sstream>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

#include <gtest/gtest.h>

template< typename Matrix, typename SymmetricMatrix >
void
copySparseToSparseMatrix_test()
{
   // Create sparse source matrix
   // clang-format off
   // /  1  2  0 |
   // |  0  5  6 |
   // \  7  0  9 /
   Matrix source( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 },
                  { 1, 1, 5 },  { 1, 2, 6 },
     { 2, 0, 7 },               { 2, 2, 9 } } );
   // clang-format on

   // Create sparse destination matrix
   Matrix destination( 3, 3 );

   // Copy matrix
   TNL::Matrices::copySparseToSparseMatrix( destination, source );

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

   Matrix destination2( 2, 4 );
   TNL::Matrices::copySparseToSparseMatrix( destination2, source2 );

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
   Matrix destination3( 2, 2 );
   TNL::Matrices::copySparseToSparseMatrix( destination3, source3 );

   EXPECT_EQ( destination3.getElement( 0, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 0, 1 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 1 ), 0.0 );

   // Test with full sparse matrix
   // clang-format off
   Matrix source4( 2, 2,
   { { 0, 0, 1 }, { 0, 1, 2 },
     { 1, 0, 3 }, { 1, 1, 4 } } );
   // clang-format on

   Matrix destination4( 2, 2 );
   TNL::Matrices::copySparseToSparseMatrix( destination4, source4 );

   EXPECT_EQ( destination4.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination4.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination4.getElement( 1, 0 ), 3.0 );
   EXPECT_EQ( destination4.getElement( 1, 1 ), 4.0 );

   // Test with symmetric sparse matrix
   // clang-format off
   // /  1  2  7 |
   // |  2  5  0 |
   // \  7  0  9 /
   Matrix source5( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 7 },
     { 1, 0, 2 }, { 1, 1, 5 },
     { 2, 0, 7 },              { 2, 2, 9 } } );
   // clang-format on

   Matrix destination5( 3, 3 );
   TNL::Matrices::copySparseToSparseMatrix( destination5, source5 );

   EXPECT_EQ( destination5.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination5.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination5.getElement( 0, 2 ), 7.0 );
   EXPECT_EQ( destination5.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( destination5.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination5.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( destination5.getElement( 2, 0 ), 7.0 );
   EXPECT_EQ( destination5.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( destination5.getElement( 2, 2 ), 9.0 );

   if constexpr( ! std::is_same_v< typename Matrix::RealType, long > ) {
      // Copy general sparse matrix to symmetric sparse matrix
      SymmetricMatrix destination6;
      TNL::Matrices::copySparseToSparseMatrix( destination6, source5 );

      EXPECT_EQ( destination6.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination6.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination6.getElement( 0, 2 ), 7.0 );
      EXPECT_EQ( destination6.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination6.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination6.getElement( 1, 2 ), 0.0 );
      EXPECT_EQ( destination6.getElement( 2, 0 ), 7.0 );
      EXPECT_EQ( destination6.getElement( 2, 1 ), 0.0 );
      EXPECT_EQ( destination6.getElement( 2, 2 ), 9.0 );

      // Copy symmetric sparse matrix to general sparse matrix
      Matrix destination7( 3, 3 );
      TNL::Matrices::copySparseToSparseMatrix( destination7, destination6 );

      EXPECT_EQ( destination7.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination7.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination7.getElement( 0, 2 ), 7.0 );
      EXPECT_EQ( destination7.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination7.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination7.getElement( 1, 2 ), 0.0 );
      EXPECT_EQ( destination7.getElement( 2, 0 ), 7.0 );
      EXPECT_EQ( destination7.getElement( 2, 1 ), 0.0 );
      EXPECT_EQ( destination7.getElement( 2, 2 ), 9.0 );
   }

   // Test with larger symmetric sparse matrix (5x5)
   // clang-format off
   // /  1  2  0  4  0 |
   // |  2  5  6  0  0 |
   // |  0  6  9  0 10 |
   // |  4  0  0 13  0 |
   // \  0  0 10  0 17 /
   Matrix source6( 5, 5,
   { { 0, 0, 1 }, { 0, 1, 2 },               { 0, 3, 4 },
     { 1, 0, 2 }, { 1, 1, 5 }, { 1, 2, 6 },
                  { 2, 1, 6 }, { 2, 2, 9 },                { 2, 4, 10 },
     { 3, 0, 4 },                            { 3, 3, 13 },
                               { 4, 2, 10 },               { 4, 4, 17 } } );
   // clang-format on

   Matrix destination8;
   TNL::Matrices::copySparseToSparseMatrix( destination8, source6 );

   EXPECT_EQ( destination8.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination8.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination8.getElement( 0, 2 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 0, 3 ), 4.0 );
   EXPECT_EQ( destination8.getElement( 0, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( destination8.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination8.getElement( 1, 2 ), 6.0 );
   EXPECT_EQ( destination8.getElement( 1, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 1, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 2, 0 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 2, 1 ), 6.0 );
   EXPECT_EQ( destination8.getElement( 2, 2 ), 9.0 );
   EXPECT_EQ( destination8.getElement( 2, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 2, 4 ), 10.0 );

   EXPECT_EQ( destination8.getElement( 3, 0 ), 4.0 );
   EXPECT_EQ( destination8.getElement( 3, 1 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 3, 2 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 3, 3 ), 13.0 );
   EXPECT_EQ( destination8.getElement( 3, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 4, 0 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 1 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 2 ), 10.0 );
   EXPECT_EQ( destination8.getElement( 4, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 4 ), 17.0 );

   if constexpr( ! std::is_same_v< typename Matrix::RealType, long > ) {
      // Copy general sparse matrix to symmetric sparse matrix
      SymmetricMatrix destination9;
      TNL::Matrices::copySparseToSparseMatrix( destination9, source6 );
      EXPECT_EQ( destination9.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination9.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination9.getElement( 0, 2 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 0, 3 ), 4.0 );
      EXPECT_EQ( destination9.getElement( 0, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination9.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination9.getElement( 1, 2 ), 6.0 );
      EXPECT_EQ( destination9.getElement( 1, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 1, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 2, 0 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 2, 1 ), 6.0 );
      EXPECT_EQ( destination9.getElement( 2, 2 ), 9.0 );
      EXPECT_EQ( destination9.getElement( 2, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 2, 4 ), 10.0 );

      EXPECT_EQ( destination9.getElement( 3, 0 ), 4.0 );
      EXPECT_EQ( destination9.getElement( 3, 1 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 3, 2 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 3, 3 ), 13.0 );
      EXPECT_EQ( destination9.getElement( 3, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 4, 0 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 1 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 2 ), 10.0 );
      EXPECT_EQ( destination9.getElement( 4, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 4 ), 17.0 );

      // Copy symmetric sparse matrix to general sparse matrix
      Matrix destination10;
      TNL::Matrices::copySparseToSparseMatrix( destination10, destination9 );
      EXPECT_EQ( destination10.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination10.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination10.getElement( 0, 2 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 0, 3 ), 4.0 );
      EXPECT_EQ( destination10.getElement( 0, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination10.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination10.getElement( 1, 2 ), 6.0 );
      EXPECT_EQ( destination10.getElement( 1, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 1, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 2, 0 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 2, 1 ), 6.0 );
      EXPECT_EQ( destination10.getElement( 2, 2 ), 9.0 );
      EXPECT_EQ( destination10.getElement( 2, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 2, 4 ), 10.0 );

      EXPECT_EQ( destination10.getElement( 3, 0 ), 4.0 );
      EXPECT_EQ( destination10.getElement( 3, 1 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 3, 2 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 3, 3 ), 13.0 );
      EXPECT_EQ( destination10.getElement( 3, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 4, 0 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 1 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 2 ), 10.0 );
      EXPECT_EQ( destination10.getElement( 4, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 4 ), 17.0 );
   }
}

template< typename SourceMatrix, typename TargetMatrix, typename SourceSymmetricMatrix, typename TargetSymmetricMatrix >
void
copySparseToSparseMatrixWithDifferentDevice_test()
{
   // Create sparse source matrix
   // clang-format off
   // /  1  2  0 |
   // |  0  5  6 |
   // \  7  0  9 /
   SourceMatrix source( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 },
                  { 1, 1, 5 },  { 1, 2, 6 },
     { 2, 0, 7 },               { 2, 2, 9 } } );
   // clang-format on

   // Create sparse destination matrix
   TargetMatrix destination( 3, 3 );

   // Copy matrix
   TNL::Matrices::copySparseToSparseMatrix( destination, source );

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
  SourceMatrix source2( 2, 4,
  { { 0, 0, 10 },               { 0, 2, 20 },
                  { 1, 1, 50 },               { 1, 3, 70 } } );
   // clang-format on

   TargetMatrix destination2( 2, 4 );
   TNL::Matrices::copySparseToSparseMatrix( destination2, source2 );

   EXPECT_EQ( destination2.getElement( 0, 0 ), 10.0 );
   EXPECT_EQ( destination2.getElement( 0, 1 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 0, 2 ), 20.0 );
   EXPECT_EQ( destination2.getElement( 0, 3 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 1 ), 50.0 );
   EXPECT_EQ( destination2.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( destination2.getElement( 1, 3 ), 70.0 );

   // Test with empty sparse matrix
   SourceMatrix source3( 2, 2 );
   TargetMatrix destination3( 2, 2 );
   TNL::Matrices::copySparseToSparseMatrix( destination3, source3 );

   EXPECT_EQ( destination3.getElement( 0, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 0, 1 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( destination3.getElement( 1, 1 ), 0.0 );

   // Test with full sparse matrix
   // clang-format off
   SourceMatrix source4( 2, 2,
   { { 0, 0, 1 }, { 0, 1, 2 },
     { 1, 0, 3 }, { 1, 1, 4 } } );
   // clang-format on

   TargetMatrix destination4( 2, 2 );
   TNL::Matrices::copySparseToSparseMatrix( destination4, source4 );

   EXPECT_EQ( destination4.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination4.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination4.getElement( 1, 0 ), 3.0 );
   EXPECT_EQ( destination4.getElement( 1, 1 ), 4.0 );

   // Test with symmetric sparse matrix
   // clang-format off
   // /  1  2  7 |
   // |  2  5  0 |
   // \  7  0  9 /
   SourceMatrix source5( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 7 },
     { 1, 0, 2 }, { 1, 1, 5 },
     { 2, 0, 7 },              { 2, 2, 9 } } );
   // clang-format on

   TargetMatrix destination5( 3, 3 );
   TNL::Matrices::copySparseToSparseMatrix( destination5, source5 );

   EXPECT_EQ( destination5.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination5.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination5.getElement( 0, 2 ), 7.0 );
   EXPECT_EQ( destination5.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( destination5.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination5.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( destination5.getElement( 2, 0 ), 7.0 );
   EXPECT_EQ( destination5.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( destination5.getElement( 2, 2 ), 9.0 );

   if constexpr( ! std::is_same_v< typename TargetSymmetricMatrix::RealType, long > ) {
      // Copy general sparse matrix to symmetric sparse matrix
      TargetSymmetricMatrix destination6;
      TNL::Matrices::copySparseToSparseMatrix( destination6, source5 );

      EXPECT_EQ( destination6.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination6.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination6.getElement( 0, 2 ), 7.0 );
      EXPECT_EQ( destination6.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination6.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination6.getElement( 1, 2 ), 0.0 );
      EXPECT_EQ( destination6.getElement( 2, 0 ), 7.0 );
      EXPECT_EQ( destination6.getElement( 2, 1 ), 0.0 );
      EXPECT_EQ( destination6.getElement( 2, 2 ), 9.0 );

      // Copy symmetric sparse matrix to general sparse matrix
      SourceMatrix destination7( 3, 3 );
      TNL::Matrices::copySparseToSparseMatrix( destination7, destination6 );

      EXPECT_EQ( destination7.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination7.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination7.getElement( 0, 2 ), 7.0 );
      EXPECT_EQ( destination7.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination7.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination7.getElement( 1, 2 ), 0.0 );
      EXPECT_EQ( destination7.getElement( 2, 0 ), 7.0 );
      EXPECT_EQ( destination7.getElement( 2, 1 ), 0.0 );
      EXPECT_EQ( destination7.getElement( 2, 2 ), 9.0 );
   }

   // Test with larger symmetric sparse matrix (5x5)
   // clang-format off
   // /  1  2  0  4  0 |
   // |  2  5  6  0  0 |
   // |  0  6  9  0 10 |
   // |  4  0  0 13  0 |
   // \  0  0 10  0 17 /
   SourceMatrix source6( 5, 5,
   { { 0, 0, 1 }, { 0, 1, 2 },               { 0, 3, 4 },
     { 1, 0, 2 }, { 1, 1, 5 }, { 1, 2, 6 },
                  { 2, 1, 6 }, { 2, 2, 9 },                { 2, 4, 10 },
     { 3, 0, 4 },                            { 3, 3, 13 },
                               { 4, 2, 10 },               { 4, 4, 17 } } );
   // clang-format on

   TargetMatrix destination8;
   TNL::Matrices::copySparseToSparseMatrix( destination8, source6 );

   EXPECT_EQ( destination8.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( destination8.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( destination8.getElement( 0, 2 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 0, 3 ), 4.0 );
   EXPECT_EQ( destination8.getElement( 0, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( destination8.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( destination8.getElement( 1, 2 ), 6.0 );
   EXPECT_EQ( destination8.getElement( 1, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 1, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 2, 0 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 2, 1 ), 6.0 );
   EXPECT_EQ( destination8.getElement( 2, 2 ), 9.0 );
   EXPECT_EQ( destination8.getElement( 2, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 2, 4 ), 10.0 );

   EXPECT_EQ( destination8.getElement( 3, 0 ), 4.0 );
   EXPECT_EQ( destination8.getElement( 3, 1 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 3, 2 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 3, 3 ), 13.0 );
   EXPECT_EQ( destination8.getElement( 3, 4 ), 0.0 );

   EXPECT_EQ( destination8.getElement( 4, 0 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 1 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 2 ), 10.0 );
   EXPECT_EQ( destination8.getElement( 4, 3 ), 0.0 );
   EXPECT_EQ( destination8.getElement( 4, 4 ), 17.0 );

   if constexpr( ! std::is_same_v< typename TargetSymmetricMatrix::RealType, long > ) {
      // Copy general sparse matrix to symmetric sparse matrix
      TargetSymmetricMatrix destination9;
      TNL::Matrices::copySparseToSparseMatrix( destination9, source6 );
      EXPECT_EQ( destination9.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination9.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination9.getElement( 0, 2 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 0, 3 ), 4.0 );
      EXPECT_EQ( destination9.getElement( 0, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination9.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination9.getElement( 1, 2 ), 6.0 );
      EXPECT_EQ( destination9.getElement( 1, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 1, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 2, 0 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 2, 1 ), 6.0 );
      EXPECT_EQ( destination9.getElement( 2, 2 ), 9.0 );
      EXPECT_EQ( destination9.getElement( 2, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 2, 4 ), 10.0 );

      EXPECT_EQ( destination9.getElement( 3, 0 ), 4.0 );
      EXPECT_EQ( destination9.getElement( 3, 1 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 3, 2 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 3, 3 ), 13.0 );
      EXPECT_EQ( destination9.getElement( 3, 4 ), 0.0 );

      EXPECT_EQ( destination9.getElement( 4, 0 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 1 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 2 ), 10.0 );
      EXPECT_EQ( destination9.getElement( 4, 3 ), 0.0 );
      EXPECT_EQ( destination9.getElement( 4, 4 ), 17.0 );

      // Copy symmetric sparse matrix to general sparse matrix
      SourceMatrix destination10;
      TNL::Matrices::copySparseToSparseMatrix( destination10, destination9 );
      EXPECT_EQ( destination10.getElement( 0, 0 ), 1.0 );
      EXPECT_EQ( destination10.getElement( 0, 1 ), 2.0 );
      EXPECT_EQ( destination10.getElement( 0, 2 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 0, 3 ), 4.0 );
      EXPECT_EQ( destination10.getElement( 0, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 1, 0 ), 2.0 );
      EXPECT_EQ( destination10.getElement( 1, 1 ), 5.0 );
      EXPECT_EQ( destination10.getElement( 1, 2 ), 6.0 );
      EXPECT_EQ( destination10.getElement( 1, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 1, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 2, 0 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 2, 1 ), 6.0 );
      EXPECT_EQ( destination10.getElement( 2, 2 ), 9.0 );
      EXPECT_EQ( destination10.getElement( 2, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 2, 4 ), 10.0 );

      EXPECT_EQ( destination10.getElement( 3, 0 ), 4.0 );
      EXPECT_EQ( destination10.getElement( 3, 1 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 3, 2 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 3, 3 ), 13.0 );
      EXPECT_EQ( destination10.getElement( 3, 4 ), 0.0 );

      EXPECT_EQ( destination10.getElement( 4, 0 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 1 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 2 ), 10.0 );
      EXPECT_EQ( destination10.getElement( 4, 3 ), 0.0 );
      EXPECT_EQ( destination10.getElement( 4, 4 ), 17.0 );
   }
}

template< typename TargetMatrix, typename SourceMatrix >
void
copySparseMatrixElementsAcrossDevicesInChunks_check( const SourceMatrix& source, bool symmetricToGeneral )
{
   using IndexType = typename SourceMatrix::IndexType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;
   using ExpectedMatrix = typename TargetMatrix::template Self< typename TargetMatrix::RealType, SourceDeviceType >;

   // The copy on the same device serves as the reference
   ExpectedMatrix expected;
   expected = source;
   TNL::Containers::Vector< IndexType, SourceDeviceType, IndexType > rowCapacities;
   expected.getRowCapacities( rowCapacities );

   for( IndexType chunkSize : { 1, 2, 3, 5, 100 } ) {
      TargetMatrix target;
      target.setDimensions( expected.getRows(), expected.getColumns() );
      target.setRowCapacities( rowCapacities );
      target.getColumnIndexes() = TNL::Matrices::paddingIndex< IndexType >;
      TNL::Matrices::detail::copySparseMatrixElementsAcrossDevices( target, source, false, symmetricToGeneral, chunkSize );

      for( IndexType rowIdx = 0; rowIdx < expected.getRows(); rowIdx++ )
         for( IndexType columnIdx = 0; columnIdx < expected.getColumns(); columnIdx++ )
            EXPECT_EQ( target.getElement( rowIdx, columnIdx ), expected.getElement( rowIdx, columnIdx ) )
               << "chunkSize = " << chunkSize << ", rowIdx = " << rowIdx << ", columnIdx = " << columnIdx;
   }
}

template< typename SourceMatrix, typename TargetMatrix, typename SourceSymmetricMatrix >
void
copySparseMatrixElementsAcrossDevicesInChunks_test()
{
   // Create sparse source matrix with empty rows, an explicit zero and a long row, so that the matrix is
   // split into chunks in various ways
   // clang-format off
   // /  1  0  0  2  0  0  0  0  0 |
   // |  0  0  0  0  0  0  0  0  0 |
   // |  3  0  4  0  0  0  0  0  5 |
   // |  6  7  8  9 10 11 12 13 14 |
   // |  0  0  0  0 15  0  0  0  0 |
   // |  0  0  0  0  0  0  0  0  0 |
   // |  0  0 16  0  0  0 17  0  0 |
   // |  0  0  0  0  0  0  0 18  0 |
   // \  0 19  0  0  0  0  0  0 20 /
   SourceMatrix source( 9, 9,
   { { 0, 0,  1 },                                         { 0, 3,  2 },
     { 2, 0,  3 }, { 2, 1,  0 }, { 2, 2,  4 },                                                                                     { 2, 8,  5 },
     { 3, 0,  6 }, { 3, 1,  7 }, { 3, 2,  8 }, { 3, 3,  9 }, { 3, 4, 10 }, { 3, 5, 11 }, { 3, 6, 12 }, { 3, 7, 13 }, { 3, 8, 14 },
                                                             { 4, 4, 15 },
                                 { 6, 2, 16 },                                           { 6, 6, 17 },
                                                                                                       { 7, 7, 18 },
                   { 8, 1, 19 },                                                                                                   { 8, 8, 20 } } );
   // clang-format on

   copySparseMatrixElementsAcrossDevicesInChunks_check< TargetMatrix >( source, false );

   if constexpr( ! std::is_same_v< typename SourceSymmetricMatrix::RealType, long > ) {
      // Reconstruct the upper part from a symmetric matrix storing the lower part of the source matrix
      SourceSymmetricMatrix symmetricSource;
      symmetricSource = source;
      copySparseMatrixElementsAcrossDevicesInChunks_check< TargetMatrix >( symmetricSource, true );
   }
}

template< typename Matrix, typename SymmetricMatrix >
void
assignmentOperator_test()
{
   // /  1  2  3 |
   // |  2  5  0 |
   // \  3  0  9 /
   SymmetricMatrix symmetric( 3, 3 );
   symmetric.setElements(
      { { 0, 0, 1 }, { 1, 0, 2 }, { 1, 1, 5 }, { 2, 0, 3 }, { 2, 2, 9 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricLower );

   // Assignment of a symmetric sparse matrix to a general sparse matrix
   // expands the upper part from the lower part.
   Matrix general( 3, 3 );
   general = symmetric;

   EXPECT_EQ( general.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( general.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( general.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( general.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( general.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( general.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( general.getElement( 2, 0 ), 3.0 );
   EXPECT_EQ( general.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( general.getElement( 2, 2 ), 9.0 );

   // Assignment of a general sparse matrix to a symmetric sparse matrix
   // copies only the lower part, i.e. the source matrix is assumed to be
   // symmetric.
   // clang-format off
   Matrix general2( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 3 },
     { 1, 0, 2 }, { 1, 1, 5 },
     { 2, 0, 3 },               { 2, 2, 9 } } );
   // clang-format on
   SymmetricMatrix symmetric2( 3, 3 );
   symmetric2 = general2;

   EXPECT_EQ( symmetric2.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( symmetric2.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( symmetric2.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( symmetric2.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( symmetric2.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( symmetric2.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( symmetric2.getElement( 2, 0 ), 3.0 );
   EXPECT_EQ( symmetric2.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( symmetric2.getElement( 2, 2 ), 9.0 );
}

template< typename SourceMatrix, typename TargetMatrix, typename SourceSymmetricMatrix, typename TargetSymmetricMatrix >
void
assignmentOperatorWithDifferentDevice_test()
{
   // /  1  2  3 |
   // |  2  5  0 |
   // \  3  0  9 /
   SourceSymmetricMatrix symmetricSource( 3, 3 );
   symmetricSource.setElements(
      { { 0, 0, 1 }, { 1, 0, 2 }, { 1, 1, 5 }, { 2, 0, 3 }, { 2, 2, 9 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricLower );

   // Assignment of a symmetric sparse matrix to a general sparse matrix
   // expands the upper part from the lower part.
   TargetMatrix generalTarget( 3, 3 );
   generalTarget = symmetricSource;

   EXPECT_EQ( generalTarget.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( generalTarget.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( generalTarget.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( generalTarget.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( generalTarget.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( generalTarget.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( generalTarget.getElement( 2, 0 ), 3.0 );
   EXPECT_EQ( generalTarget.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( generalTarget.getElement( 2, 2 ), 9.0 );

   // Assignment of a general sparse matrix to a symmetric sparse matrix
   // copies only the lower part, i.e. the source matrix is assumed to be
   // symmetric.
   // clang-format off
   SourceMatrix generalSource( 3, 3,
   { { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 3 },
     { 1, 0, 2 }, { 1, 1, 5 },
     { 2, 0, 3 },               { 2, 2, 9 } } );
   // clang-format on
   TargetSymmetricMatrix symmetricTarget( 3, 3 );
   symmetricTarget = generalSource;

   EXPECT_EQ( symmetricTarget.getElement( 0, 0 ), 1.0 );
   EXPECT_EQ( symmetricTarget.getElement( 0, 1 ), 2.0 );
   EXPECT_EQ( symmetricTarget.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( symmetricTarget.getElement( 1, 0 ), 2.0 );
   EXPECT_EQ( symmetricTarget.getElement( 1, 1 ), 5.0 );
   EXPECT_EQ( symmetricTarget.getElement( 1, 2 ), 0.0 );
   EXPECT_EQ( symmetricTarget.getElement( 2, 0 ), 3.0 );
   EXPECT_EQ( symmetricTarget.getElement( 2, 1 ), 0.0 );
   EXPECT_EQ( symmetricTarget.getElement( 2, 2 ), 9.0 );
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
