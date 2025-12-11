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
