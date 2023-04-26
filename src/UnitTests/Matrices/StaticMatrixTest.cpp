#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Matrices/StaticMatrix.h>

using namespace TNL::Containers;
using namespace TNL::Matrices;

TEST( StaticNDArrayTest, 3x4_row_major )
{
   constexpr int I = 3, J = 4;
   StaticMatrix< int, I, J > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 4x3_row_major )
{
   constexpr int I = 4, J = 3;
   StaticMatrix< int, I, J > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 3x4_column_major )
{
   constexpr int I = 3, J = 4;
   using Permutation = std::index_sequence< 1, 0 >;
   StaticMatrix< int, I, J, Permutation > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 4x3_column_major )
{
   constexpr int I = 4, J = 3;
   using Permutation = std::index_sequence< 1, 0 >;
   StaticMatrix< int, I, J, Permutation > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, set_elements )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = 1.2f;

   EXPECT_EQ( M( 0, 0 ), 1.2f );
   EXPECT_EQ( M( 0, 1 ), 1.2f );
   EXPECT_EQ( M( 0, 2 ), 1.2f );
   EXPECT_EQ( M( 1, 0 ), 1.2f );
   EXPECT_EQ( M( 1, 1 ), 1.2f );
   EXPECT_EQ( M( 1, 2 ), 1.2f );
   EXPECT_EQ( M( 2, 0 ), 1.2f );
   EXPECT_EQ( M( 2, 1 ), 1.2f );
   EXPECT_EQ( M( 2, 2 ), 1.2f );
}

TEST( StaticNDArrayTest, matrix_addition )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = 1.2f;
   StaticMatrix< float, I, J, Permutation > B = 1.3f;
   StaticMatrix< float, I, J, Permutation > C = A + B;

   EXPECT_EQ( C( 0, 0 ), 2.5f );
   EXPECT_EQ( C( 0, 1 ), 2.5f );
   EXPECT_EQ( C( 0, 2 ), 2.5f );
   EXPECT_EQ( C( 1, 0 ), 2.5f );
   EXPECT_EQ( C( 1, 1 ), 2.5f );
   EXPECT_EQ( C( 1, 2 ), 2.5f );
   EXPECT_EQ( C( 2, 0 ), 2.5f );
   EXPECT_EQ( C( 2, 1 ), 2.5f );
   EXPECT_EQ( C( 2, 2 ), 2.5f );
}

TEST( StaticNDArrayTest, matrix_substraction )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = 3.f;
   StaticMatrix< float, I, J, Permutation > B = 1.5f;
   StaticMatrix< float, I, J, Permutation > C = A - B;

   EXPECT_EQ( C( 0, 0 ), 1.5f );
   EXPECT_EQ( C( 0, 1 ), 1.5f );
   EXPECT_EQ( C( 0, 2 ), 1.5f );
   EXPECT_EQ( C( 1, 0 ), 1.5f );
   EXPECT_EQ( C( 1, 1 ), 1.5f );
   EXPECT_EQ( C( 1, 2 ), 1.5f );
   EXPECT_EQ( C( 2, 0 ), 1.5f );
   EXPECT_EQ( C( 2, 1 ), 1.5f );
   EXPECT_EQ( C( 2, 2 ), 1.5f );
}

TEST( StaticNDArrayTest, matrix_expression )
{
   constexpr int I = 2, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = 3.f;
   StaticMatrix< float, I, J, Permutation > B = 1.5f;
   StaticMatrix< float, I, J, Permutation > C = 2.3f * A - B * 3.6 + 3 * A - A / 8;

   EXPECT_EQ( C( 0, 0 ), 10.125f );
   EXPECT_EQ( C( 0, 1 ), 10.125f );
   EXPECT_EQ( C( 0, 2 ), 10.125f );
   EXPECT_EQ( C( 1, 0 ), 10.125f );
   EXPECT_EQ( C( 1, 1 ), 10.125f );
   EXPECT_EQ( C( 1, 2 ), 10.125f );
}

TEST( StaticNDArrayTest, 3x3_times_3x3_matrix_multiplication )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = { 3.f, 8.f, 6.f, 4.f, 1.f, 2.f, 3.5f, 8.f, 1.f };
   StaticMatrix< float, I, J, Permutation > B = { 2.f, 6.f, 1.2f, 9.f, 7.3f, 5.f, 4.f, 6.f, 2.f };

   StaticMatrix< float, I, J, Permutation > C = A * B;

   EXPECT_EQ( C( 0, 0 ), 102.f );
   EXPECT_EQ( C( 0, 1 ), 112.4f );
   EXPECT_EQ( C( 0, 2 ), 55.6f );
   EXPECT_EQ( C( 1, 0 ), 25.f );
   EXPECT_EQ( C( 1, 1 ), 43.3f );
   EXPECT_EQ( C( 1, 2 ), 13.8f );
   EXPECT_EQ( C( 2, 0 ), 83.f );
   EXPECT_EQ( C( 2, 1 ), 85.4f );
   EXPECT_EQ( C( 2, 2 ), 46.2f );

   StaticMatrix< float, I, J, Permutation > D = B * A;

   EXPECT_EQ( D( 0, 0 ), 34.2f );
   EXPECT_EQ( D( 0, 1 ), 31.6f );
   EXPECT_EQ( D( 0, 2 ), 25.2f );
   EXPECT_EQ( D( 1, 0 ), 73.7f );
   EXPECT_EQ( D( 1, 1 ), 119.3f );
   EXPECT_EQ( D( 1, 2 ), 73.6f );
   EXPECT_EQ( D( 2, 0 ), 43.f );
   EXPECT_EQ( D( 2, 1 ), 54.f );
   EXPECT_EQ( D( 2, 2 ), 38.f );
}

TEST( StaticNDArrayTest, 2x3_times_3x4_matrix_multiplication )
{
   constexpr int I1 = 2, J1 = 3;
   constexpr int I2 = 3, J2 = 4;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I1, J1, Permutation > A = { 3.f, 8.f, 6.f, 4.f, 1.f, 2.f };
   StaticMatrix< float, I2, J2, Permutation > B = { 2.f, 6.f, 1.2f, 1.f, 9.f, 7.3f, 5.f, 2.f, 4.f, 6.f, 2.f, 3.f };

   StaticMatrix< float, I1, J2, Permutation > C = A * B;

   EXPECT_EQ( C( 0, 0 ), 102.f );
   EXPECT_EQ( C( 0, 1 ), 112.4f );
   EXPECT_EQ( C( 0, 2 ), 55.6f );
   EXPECT_EQ( C( 0, 3 ), 37.f );
   EXPECT_EQ( C( 1, 0 ), 25.f );
   EXPECT_EQ( C( 1, 1 ), 43.3f );
   EXPECT_EQ( C( 1, 2 ), 13.8f );
   EXPECT_EQ( C( 1, 3 ), 12.f );
}

TEST( StaticNDArrayTest, 3x3_transpose )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = { 3.f, 8.f, 6.f, 4.f, 1.f, 2.f, 3.5f, 8.f, 1.f };

   A = transpose( A );

   EXPECT_EQ( A( 0, 0 ), 3.f );
   EXPECT_EQ( A( 0, 1 ), 4.f );
   EXPECT_EQ( A( 0, 2 ), 3.5f );
   EXPECT_EQ( A( 1, 0 ), 8.f );
   EXPECT_EQ( A( 1, 1 ), 1.f );
   EXPECT_EQ( A( 1, 2 ), 8.f );
   EXPECT_EQ( A( 2, 0 ), 6.f );
   EXPECT_EQ( A( 2, 1 ), 2.f );
   EXPECT_EQ( A( 2, 2 ), 1.f );
}

TEST( StaticNDArrayTest, 2x3_transpose )
{
   constexpr int I = 2, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > A = { 3.f, 8.f, 6.f, 4.f, 1.f, 2.f };

   StaticMatrix< float, J, I, Permutation > B = transpose( A );

   EXPECT_EQ( B( 0, 0 ), 3.f );
   EXPECT_EQ( B( 0, 1 ), 4.f );
   EXPECT_EQ( B( 1, 0 ), 8.f );
   EXPECT_EQ( B( 1, 1 ), 1.f );
   EXPECT_EQ( B( 2, 0 ), 6.f );
   EXPECT_EQ( B( 2, 1 ), 2.f );
}

TEST( StaticNDArrayTest, 2x2_determinant )
{
   constexpr int I = 2, J = 2;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 4, 1 };

   float det = determinant( M );

   EXPECT_EQ( -29, det );
}

TEST( StaticNDArrayTest, 3x3_determinant )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 6, 4, 1, 2, 3.5, 7, 1 };

   float det = determinant( M );

   EXPECT_EQ( 132, det );
}

TEST( StaticNDArrayTest, 4x4_determinant )
{
   constexpr int I = 4, J = 4;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 6, 2, 4, 1, 2, 1.3, 3.5, 7, 1, 5, 4, 2, 2.5, 7 };

   float det = determinant( M );

   EXPECT_EQ( 723.25, det );
}

TEST( StaticNDArrayTest, 2x2_linear_system )
{
   constexpr int I = 2, J = 2;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 4, 1 };
   StaticVector< I, float > b( 2, 4 );

   StaticVector< I, float > solution = solve( M, b );
   StaticVector< I, float > exact_solution = { 1.03448, -0.13793 };

   EXPECT_NEAR( exact_solution[ 0 ], solution[ 0 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 1 ], solution[ 1 ], 1.0e-5 );
}

TEST( StaticNDArrayTest, 3x3_linear_system )
{
   constexpr int I = 3, J = 3;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 6, 4, 1, 2, 3.5, 7, 1 };
   StaticVector< I, float > b( 2, 3, 11.6 );

   StaticVector< I, float > solution = solve( M, b );
   StaticVector< I, float > exact_solution = { 1.45455, 1.21818, -2.01818 };

   EXPECT_NEAR( exact_solution[ 0 ], solution[ 0 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 1 ], solution[ 1 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 2 ], solution[ 2 ], 1.0e-5 );
}

TEST( StaticNDArrayTest, 4x4_linear_system )
{
   constexpr int I = 4, J = 4;
   using Permutation = std::index_sequence< 0, 1 >;

   StaticMatrix< float, I, J, Permutation > M = { 3, 8, 6, 2, 4, 1, 2, 1.3, 3.5, 7, 1, 5, 4, 2, 2.5, 7 };
   StaticVector< I, float > b( 2, 3, 11.6, 7.1 );

   StaticVector< I, float > solution = solve( M, b );
   StaticVector< I, float > exact_solution = { 1.08765, 0.827611, -1.55068, 0.710128 };

   EXPECT_NEAR( exact_solution[ 0 ], solution[ 0 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 1 ], solution[ 1 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 2 ], solution[ 2 ], 1.0e-5 );
   EXPECT_NEAR( exact_solution[ 3 ], solution[ 3 ], 1.0e-5 );
}

#endif // HAVE_GTEST

#include "../main.h"
