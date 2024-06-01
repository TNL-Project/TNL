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
